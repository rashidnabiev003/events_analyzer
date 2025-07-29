# src/search/semantic_search.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable, Optional, Dict

import json
import math
import numpy as np
import pandas as pd
import torch

try:
    import faiss  # faiss-cpu или faiss-gpu, автодетект ниже
except Exception as e:
    raise RuntimeError("FAISS is required (faiss-cpu или faiss-gpu).") from e

from sentence_transformers import SentenceTransformer

# Опционально для реранкинга
try:
    from FlagEmbedding import BGEM3FlagModel
    _HAS_BGE_RERANKER = True
except Exception:
    _HAS_BGE_RERANKER = False


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _to_gpu_if_available(index: faiss.Index, gpu_id: int = 0, force_cpu: bool = False) -> faiss.Index:
    if force_cpu:
        return index
    try:
        if hasattr(faiss, "StandardGpuResources"):
            num_gpus = getattr(faiss, "get_num_gpus", lambda: 0)()
            if num_gpus > 0 and torch.cuda.is_available():
                res = faiss.StandardGpuResources()  # стандартный пул ресурсов
                index = faiss.index_cpu_to_gpu(res, gpu_id, index)
        return index
    except Exception:
        # Любая ошибка — тихий откат на CPU
        return index


@dataclass
class BuildConfig:
    model_name: str = "BAAI/bge-m3"  # sentence-transformers интерфейс
    batch_size: int = 64
    use_fp16: bool = True
    force_cpu: bool = False


class EventsSemanticSearch:
    """
    Простой и надёжный поисковик «похожих мероприятий»:
    - Flat IP (точный поиск), автопереключение CPU/GPU
    - Косинусная близость через нормализацию
    - Опциональный реранкинг BGE-M3 (dense+sparse+colbert)
    """

    def __init__(self, workdir: Path, cfg: Optional[BuildConfig] = None, enable_rerank: bool = True):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or BuildConfig()
        self.enable_rerank = enable_rerank and _HAS_BGE_RERANKER

        # Файлы индекса/метаданных
        self.index_path = self.workdir / "events.faiss"
        self.meta_path = self.workdir / "metadata.json"
        self.embed_path = self.workdir / "embeddings.npy"

        # Ленивая инициализация
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self.embedder: Optional[SentenceTransformer] = None
        self.reranker: Optional[BGEM3FlagModel] = None


    def build_from_dataframe(self, df: pd.DataFrame, text_col: str = "raw_text", id_col: str = "event_id") -> None:
        """
        Строит FAISS-индекс по DataFrame c колонками: id_col, text_col.
        Сохраняет: events.faiss, metadata.json, embeddings.npy
        """
        texts = df[text_col].astype(str).fillna("").tolist()
        ids = df[id_col].astype(str).tolist()

        self._load_embedder()
        emb = self._encode_batch(texts)
        emb = _l2_normalize(emb).astype("float32")  # для IP=cosine

        dim = emb.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)  # точный exhaust. поиск (поддерживается на GPU)
        cpu_index.add(emb)
        self.index = _to_gpu_if_available(cpu_index, force_cpu=self.cfg.force_cpu)

        # метаданные
        self.metadata = [{"event_id": i, "text": t} for i, t in zip(ids, texts)]

        # persist
        faiss.write_index(cpu_index, str(self.index_path))  # сохраняем CPU-вариант для переносимости
        np.save(self.embed_path, emb)
        self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        """Загружает индекс и метаданные (авто CPU/GPU)."""
        cpu_index = faiss.read_index(str(self.index_path))
        self.index = _to_gpu_if_available(cpu_index, force_cpu=self.cfg.force_cpu)
        self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int = 50, rerank_top_m: int = 100) -> List[Tuple[Dict, float]]:
        """
        Возвращает список (metadata, score) по убыванию.
        Сначала FAISS (быстро), опционально — реранкинг BGE-M3 для top-M.
        """
        assert self.index is not None, "index is not loaded"
        self._load_embedder()

        q = self._encode_batch([query])
        q = _l2_normalize(q).astype("float32")
        D, I = self.index.search(q, max(top_k, rerank_top_m))
        idxs = [i for i in I[0] if i != -1]
        faiss_candidates = [(self.metadata[i], float(D[0][k])) for k, i in enumerate(idxs)]

        if not self.enable_rerank or len(faiss_candidates) <= 1:
            return faiss_candidates[:top_k]

        # Реранкинг BGE-M3: compute_score -> "colbert+sparse+dense"
        self._load_reranker()
        pairs = [[query, m["text"]] for (m, _) in faiss_candidates[:rerank_top_m]]
        # compute_score: официальная функция для BGE-M3
        scores = self.reranker.compute_score(pairs)["colbert+sparse+dense"]
        reranked = list(zip([c[0] for c in faiss_candidates[:rerank_top_m]], scores))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def similar_pairs_for_all(self, df: pd.DataFrame, text_col: str = "raw_text",
                              id_col: str = "event_id", top_k_per_event: int = 20) -> List[Tuple[str, str]]:
        """
        Генерирует пары (A_id, B_id) для всех мероприятий — top-K похожих для каждого A.
        Удобно для сокращения матрицы рисков.
        """
        pairs: List[Tuple[str, str]] = []
        for _, row in df.iterrows():
            a_id = str(row[id_col])
            a_text = str(row[text_col])
            results = self.search(a_text, top_k=top_k_per_event)
            for meta, _ in results:
                b_id = meta["event_id"]
                if b_id != a_id:
                    pairs.append((a_id, b_id))
        return pairs

    def _load_embedder(self):
        if self.embedder is None:
            device = "cuda" if (torch.cuda.is_available() and not self.cfg.force_cpu) else "cpu"
            self.embedder = SentenceTransformer(self.cfg.model_name, device=device)

    def _load_reranker(self):
        if (self.reranker is None) and self.enable_rerank:
            device = "cuda" if (torch.cuda.is_available() and not self.cfg.force_cpu) else "cpu"
            # FP16 ускоряет инференс на GPU
            self.reranker = BGEM3FlagModel(self.cfg.model_name, use_fp16=(device == "cuda"), device=device)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        all_emb: List[np.ndarray] = []
        bs = self.cfg.batch_size
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            with torch.inference_mode():
                vec = self.embedder.encode(chunk, convert_to_numpy=True, show_progress_bar=False,
                                           normalize_embeddings=False)  # нормализуем сами
                all_emb.append(vec.astype("float32"))
        return np.vstack(all_emb)
