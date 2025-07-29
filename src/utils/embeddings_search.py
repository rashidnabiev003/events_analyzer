# src/search/semantic_search.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import json
import numpy as np
import pandas as pd
import torch

try:
    import faiss  # faiss-cpu или faiss-gpu
except Exception as e:
    raise RuntimeError("FAISS is required (pip install faiss-cpu или faiss-gpu).") from e


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


@dataclass
class BuildConfig:
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 64
    force_cpu: bool = False  # True — всё на CPU (без конкуренции за VRAM)


class EventsSemanticSearch:
    """
    Простой продакшен-вариант:
    - эмбеддинги BGE-M3 через sentence-transformers
    - FAISS IndexFlatIP (косинус через нормализацию)
    - батчевый поиск Top-K для всех мероприятий за один вызов
    - без реранкинга (максимум скорости)
    """
    def __init__(self, workdir: Path, cfg: Optional[BuildConfig] = None):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or BuildConfig()

        self.index_path = self.workdir / "events.faiss"
        self.meta_path = self.workdir / "metadata.json"
        self.embed_path = self.workdir / "embeddings.npy"

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self.embedder = None

    # ---------- public ----------

    def build_from_dataframe(self, df: pd.DataFrame, text_col: str = "raw_text", id_col: str = "event_id") -> None:
        texts = df[text_col].astype(str).fillna("").tolist()
        ids = df[id_col].astype(str).tolist()

        self._load_embedder()
        emb = self._encode_batch(texts)            # (N, d)
        emb = _l2_normalize(emb).astype("float32") # для косинусной близости через IP

        dim = emb.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(emb)
        self.index = cpu_index  # держим CPU-вариант; перенос на GPU можно добавить при необходимости

        self.metadata = [{"event_id": i, "text": t} for i, t in zip(ids, texts)]

        faiss.write_index(cpu_index, str(self.index_path))
        np.save(self.embed_path, emb)
        self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        cpu_index = faiss.read_index(str(self.index_path))
        self.index = cpu_index
        self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def batch_similar_pairs_for_all(self, top_k_per_event: int = 20, id_col: str = "event_id") -> List[Tuple[str, str]]:
        """
        Возвращает пары (A_id, B_id) для всех мероприятий: Top-K похожих для каждого A.
        Делается одним батчевым поиском FAISS по матрице эмбеддингов (очень быстро).
        """
        assert self.index is not None, "index is not loaded"
        emb = np.load(self.embed_path)  # (N, d), уже нормализованные
        D, I = self.index.search(emb, top_k_per_event + 1)  # +1 — чтобы потом убрать self-match

        pairs: List[Tuple[str, str]] = []
        for a_idx, neighs in enumerate(I):
            a_id = self.metadata[a_idx][id_col]
            for b_idx in neighs:
                if b_idx == -1 or b_idx == a_idx:
                    continue
                b_id = self.metadata[b_idx][id_col]
                pairs.append((str(a_id), str(b_id)))
        return pairs


    def _load_embedder(self):
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            device = "cpu" if self.cfg.force_cpu or not torch.cuda.is_available() else "cuda"
            self.embedder = SentenceTransformer(self.cfg.model_name, device=device)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        all_emb: List[np.ndarray] = []
        bs = self.cfg.batch_size
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            vec = self.embedder.encode(chunk, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
            all_emb.append(vec.astype("float32"))
        return np.vstack(all_emb)
