"""
First-stage retrieval for benchmarks that ship no candidate pools.

  bm25     BM25 top-k (rank_bm25).
  hybrid   BM25 + SPLADE + mpnet, fused by reciprocal rank fusion (RRF, k=60), the
           pool construction used for the paper's table benchmarks.

With `inject_gold`, any relevant document missing from a pool is appended so that
reranking quality, not first-stage recall, is what is measured.
"""

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

MPNET_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
SPLADE_MODEL_ID = "naver/splade-v3"
RRF_K = 60

Ranked = List[Tuple[str, float]]


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _top(scores, ids: List[str], k: int) -> Ranked:
    import numpy as np

    n = min(k, len(scores))
    idx = np.argpartition(scores, -n)[-n:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(ids[i], float(scores[i])) for i in idx]


def rrf_merge(ranked_lists: Sequence[Ranked], top_k: int, k: int = RRF_K) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, 1):
            scores[doc_id] += 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]]


class BM25Index:
    def __init__(self, ids: List[str], corpus: Dict[str, str]):
        from rank_bm25 import BM25Okapi

        self.ids = ids
        self.index = BM25Okapi([_tokenize(corpus[d]) for d in ids])

    def search(self, query: str, top_k: int) -> Ranked:
        return _top(self.index.get_scores(_tokenize(query)), self.ids, top_k)


class SpladeIndex:
    def __init__(self, ids: List[str], texts: List[str], device: str):
        import scipy.sparse as sp
        from sentence_transformers import SparseEncoder

        self.ids, self.model = ids, SparseEncoder(SPLADE_MODEL_ID, device=device)
        emb = self.model.encode_document(texts, batch_size=128, show_progress_bar=True)
        dense = emb.to_dense() if emb.is_sparse else emb
        self.matrix = sp.csr_matrix(dense.cpu().float().numpy())

    def search(self, query: str, top_k: int) -> Ranked:
        import scipy.sparse as sp

        emb = self.model.encode_query([query], show_progress_bar=False)
        dense = emb.to_dense() if emb.is_sparse else emb
        qvec = sp.csr_matrix(dense.cpu().float().numpy())
        return _top((self.matrix @ qvec.T).toarray().flatten(), self.ids, top_k)


class DenseIndex:
    def __init__(self, ids: List[str], texts: List[str], device: str):
        from sentence_transformers import SentenceTransformer

        self.ids, self.model = ids, SentenceTransformer(MPNET_MODEL_ID, device=device)
        self.matrix = self.model.encode(texts, batch_size=256, show_progress_bar=True,
                                        normalize_embeddings=True, convert_to_numpy=True)

    def search(self, query: str, top_k: int) -> Ranked:
        qvec = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return _top((self.matrix @ qvec).flatten(), self.ids, top_k)


def build_pools(corpus: Dict[str, str], queries: Dict[str, str], gold: Dict[str, Sequence[str]],
                kind: str = "bm25", top_k: int = 100, inject_gold: bool = False,
                device: str = "cuda:0", skip_splade: bool = False) -> Dict[str, List[str]]:
    """Candidate pool (best first) for every query id in `queries`."""
    if kind not in ("bm25", "hybrid"):
        raise ValueError(f"Unknown first stage {kind!r}; expected 'bm25' or 'hybrid'")
    ids = list(corpus)
    indexes = [BM25Index(ids, corpus)]
    if kind == "hybrid":
        texts = [corpus[d] for d in ids]
        if not skip_splade:
            try:
                indexes.append(SpladeIndex(ids, texts, device))
            except Exception as e:  # SPLADE is optional: fall back to BM25 + dense
                print(f"[retrieval] SPLADE unavailable ({e}); continuing without it")
        indexes.append(DenseIndex(ids, texts, device))

    pools: Dict[str, List[str]] = {}
    for qid, query in queries.items():
        rankings = [index.search(query, top_k) for index in indexes]
        pool = [d for d, _ in rankings[0]] if kind == "bm25" else rrf_merge(rankings, top_k)
        if inject_gold:
            present = set(pool)
            pool = pool + [g for g in gold.get(qid, ()) if g not in present]
        pools[qid] = pool
    return pools
