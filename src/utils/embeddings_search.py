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
	from FlagEmbedding import BGEM3FlagModel
	_HAS_BGE = True
except Exception:
	_HAS_BGE = False

# Defer FAISS import errors to runtime methods instead of module import time
_faiss = None

def _require_faiss():
	global _faiss
	if _faiss is None:
		try:
			import faiss as _faiss_mod  # type: ignore
			_faiss = _faiss_mod
		except Exception as e:  # pragma: no cover
			raise RuntimeError("FAISS is required (pip install faiss-cpu или faiss-gpu).") from e
	return _faiss


def _l2_normalize(x: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
	return x / norms


@dataclass
class BuildConfig:
	model_name: str = "BAAI/bge-m3"
	batch_size: int = 64
	force_cpu: bool = False          # эмбеддер на CPU
	enable_rerank: bool = True       # включать реранкер
	rerank_device: str = "cpu"       # "cuda" или "cpu"
	rerank_batch_size: int = 2048    # размер батча пар для compute_score
	use_fp16_rerank: bool = True     # FP16 на GPU

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

		self.index: Optional[object] = None
		self.metadata: List[Dict] = []
		self.embedder = None
		self.reranker = None
	
	def _load_reranker(self):
		if not (self.cfg.enable_rerank and _HAS_BGE):
			return
		if self.reranker is None:
			device = self.cfg.rerank_device
			use_fp16 = self.cfg.use_fp16_rerank and (device.startswith("cuda"))
			self.reranker = BGEM3FlagModel(self.cfg.model_name, use_fp16=use_fp16, device=device)

	# ---------- public ----------

	def build_from_dataframe(self, df: pd.DataFrame, text_col: str = "raw_text", id_col: str = "event_id") -> None:
		texts = df[text_col].astype(str).fillna("").tolist()
		ids = df[id_col].astype(str).tolist()

		self._load_embedder()
		emb = self._encode_batch(texts)            # (N, d)
		emb = _l2_normalize(emb).astype("float32") # для косинусной близости через IP

		faiss = _require_faiss()
		dim = emb.shape[1]
		cpu_index = faiss.IndexFlatIP(dim)
		cpu_index.add(emb)
		self.index = cpu_index  # держим CPU-вариант; перенос на GPU можно добавить при необходимости

		self.metadata = [{"event_id": i, "text": t} for i, t in zip(ids, texts)]

		faiss.write_index(cpu_index, str(self.index_path))
		np.save(self.embed_path, emb)
		self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")

	def load(self) -> None:
		faiss = _require_faiss()
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
		faiss = _require_faiss()
		D, I = faiss.Index.search(self.index, emb, top_k_per_event + 1) if hasattr(faiss, 'Index') else self.index.search(emb, top_k_per_event + 1)  # type: ignore[attr-defined]

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
			from sentence_transformers import SentenceTransformer  # local import to avoid import-time failure
			device = "cpu" if self.cfg.force_cpu or not torch.cuda.is_available() else "cuda"
			self.embedder = SentenceTransformer(self.cfg.model_name, device=device)

	def _encode_batch(self, texts: List[str]) -> np.ndarray:
		all_emb: List[np.ndarray] = []
		bs = self.cfg.batch_size
		for i in range(0, len(texts), bs):
			chunk = texts[i:i+bs]
			vec = self.embedder.encode(
				chunk,
				convert_to_numpy=True,
				show_progress_bar=False,
				normalize_embeddings=False,
			)
			all_emb.append(vec.astype("float32"))
		return np.vstack(all_emb)
	
	def make_pairs_percent(
		self,
		k_preselect: int = 50,             # сколько соседей на событие взять от FAISS
		min_faiss_sim: float = 0.20,       # отсечь слабых ещё ДО реранка
		sim_threshold: float | None = 0.5, # порог «по проценту схожести» (0..1); если None — используем keep_top_pct
		keep_top_pct: float | None = None, # либо доля лучших на событие (например 0.3=30%); если None — используем sim_threshold
		per_event_cap: int | None = None,  # жёсткий предел пар на событие после фильтров
		id_col: str = "event_id",
		dedup_bidirectional: bool = True,
		use_reranker: bool = True,         # включить реранкер
		metric_key: str = "colbert+sparse+dense",
	) -> list[tuple[str, str]]:
		"""
		1) FAISS батчем: берём k_preselect соседей на событие (+1 self-match).
		2) Фильтруем по min_faiss_sim.
		3) (опц.) Батчевый реранк BGE-M3 compute_score.
		4) Отбираем по sim_threshold ИЛИ keep_top_pct (на событие).
		5) (опц.) dedup (A,B)==(B,A), cap на событие.
		"""
		assert self.index is not None, "index is not loaded"
		emb = np.load(self.embed_path)
		# FAISS: cosine sim, т.к. эмбеддинги L2-нормализованы
		faiss = _require_faiss()
		D, I = self.index.search(emb, k_preselect + 1)  # type: ignore[attr-defined]

		# 1) Собираем кандидатов
		pairs_idx: list[tuple[int, int]] = []
		pairs_sim_faiss: list[float] = []
		for a_idx, neighs in enumerate(I):
			for rank, b_idx in enumerate(neighs):
				if b_idx == -1 or b_idx == a_idx:
					continue
				sim = float(D[a_idx, rank])
				if sim < min_faiss_sim:
					continue
				pairs_idx.append((a_idx, b_idx))
				pairs_sim_faiss.append(sim)

		if not pairs_idx:
			return []

		# 2) Батчевый реранкинг (опционально)
		rerank_scores: list[float] | None = None
		if use_reranker and self.cfg.enable_rerank and _HAS_BGE:
			self._load_reranker()
			pairs_text = [[self.metadata[a]["text"], self.metadata[b]["text"]] for a, b in pairs_idx]
			rerank_scores = []
			bs = self.cfg.rerank_batch_size
			for i in range(0, len(pairs_text), bs):
				chunk = pairs_text[i:i+bs]
				out = self.reranker.compute_score(chunk)
				scores = out.get(metric_key) or out.get("colbert+sparse+dense")
				if isinstance(scores, list):
					rerank_scores.extend([float(s) for s in scores])
				else:
					# на всякий случай
					rerank_scores.extend([float(scores)] * len(chunk))

		# 3) Группируем по A, применяем порог/перцентиль
		from collections import defaultdict
		by_a: dict[int, list[tuple[int, float]]] = defaultdict(list)  # a_idx -> [(b_idx, score), ...]
		for k, (a_idx, b_idx) in enumerate(pairs_idx):
			score = rerank_scores[k] if rerank_scores is not None else pairs_sim_faiss[k]
			by_a[a_idx].append((b_idx, score))

		result_pairs: list[tuple[str, str]] = []
		seen: set[tuple[str, str]] = set()

		for a_idx, items in by_a.items():
			# сортировка по score убыв.
			items.sort(key=lambda x: x[1], reverse=True)

			# выборка по порогу или по перцентилю
			selected: list[tuple[int, float]]
			if sim_threshold is not None:
				selected = [(b, s) for (b, s) in items if s >= sim_threshold]
				if per_event_cap is not None:
					selected = selected[:per_event_cap]
			else:
				assert keep_top_pct is not None and 0 < keep_top_pct <= 1
				k_keep = max(1, int(round(len(items) * keep_top_pct)))
				if per_event_cap is not None:
					k_keep = min(k_keep, per_event_cap)
				selected = items[:k_keep]

			a_id = str(self.metadata[a_idx][id_col])
			for b_idx, _ in selected:
				b_id = str(self.metadata[b_idx][id_col])
				if dedup_bidirectional:
					key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
					if key in seen:
						continue
					seen.add(key)
				result_pairs.append((a_id, b_id))

		return result_pairs

