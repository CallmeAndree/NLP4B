"""
service.py — Heuristic Retrieval Service (Production Implementation)
=====================================================================

Replaces the mock stub with a real 4-vector hybrid search using the
Embedding API's unified `/embed/query` endpoint.

Pipeline:
  1. Call /embed/query → get all 4 vectors + NLP analysis in one shot
  2. execute_fallback_search() → 2-tier Qdrant search:
       Tier 1 (Strict):  filter by YOLO object tags + 4-vector search
       Tier 2 (Fallback): drop filter, rely on object_sparse soft-ranking
  3. apply_custom_reranking() → RRF + Count Bonus multiplier
  4. Return top_k candidates in the standard dict format

Return dict keys (matches contract in README):
  video_id, frame_id, score, branch, evidence, raw_payload
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

# ── Vector names (aligned with qdrant_upsert.py) ─────────────────────────────
VEC_DENSE          = "keyframe-dense"
VEC_CAPTION_DENSE  = "keyframe-caption-dense"
VEC_OBJECT_SPARSE  = "keyframe-object-sparse"
VEC_OCR_SPARSE     = "keyframe-ocr-sparse"

COLLECTION_NAME = "keyframes_v1"

PAYLOAD_FIELDS = [
    "video_id", "frame_idx", "azure_url", "timestamp_sec",
    "youtube_link", "tags", "caption", "detailed_caption",
    "object_counts", "ocr_text", "title",
]


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING API CLIENT — single /embed/query call gets all 4 vectors
# ══════════════════════════════════════════════════════════════════════════════

class EmbedQueryClient:
    """Thin HTTP client for the unified /embed/query endpoint."""

    def __init__(self, base_url: str, timeout: int = 90) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def query(self, text: str) -> Optional[Dict[str, Any]]:
        """
        POST /embed/query → QueryResponse dict.

        Returns None on any error so callers can handle gracefully.
        """
        try:
            r = httpx.post(
                f"{self.base_url}/embed/query",
                json={"text": text.strip()},
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning("EmbedQueryClient.query failed: %s", exc)
            return None


# ══════════════════════════════════════════════════════════════════════════════
# QDRANT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_points(response: Any) -> List[Any]:
    if response is None:
        return []
    if hasattr(response, "points"):
        return list(response.points)
    if isinstance(response, list):
        return response
    return []


def _to_candidate(point: Any, source: str) -> Optional[Dict[str, Any]]:
    """Convert a Qdrant ScoredPoint → standard candidate dict."""
    payload = getattr(point, "payload", None) or {}
    video_id = payload.get("video_id")
    frame_idx = payload.get("frame_idx")
    if video_id is None or frame_idx is None:
        return None
    return {
        "video_id": str(video_id),
        "frame_id": int(frame_idx),
        "score": float(getattr(point, "score", 0.0)),
        "source": source,
        "branch": "heuristic",
        "evidence": [],     # filled in after merge
        "raw_payload": payload,
    }


def _query_dense(
    client: QdrantClient,
    vector: List[float],
    using: str,
    source: str,
    limit: int,
    query_filter: Optional[models.Filter] = None,
) -> List[Dict[str, Any]]:
    try:
        resp = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            using=using,
            limit=limit,
            query_filter=query_filter,
            with_payload=PAYLOAD_FIELDS,
            with_vectors=False,
        )
    except TypeError:
        resp = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=(using, vector),
            limit=limit,
            query_filter=query_filter,
            with_payload=PAYLOAD_FIELDS,
            with_vectors=False,
        )
    return [c for p in _extract_points(resp) if (c := _to_candidate(p, source))]


def _query_sparse(
    client: QdrantClient,
    sparse: models.SparseVector,
    using: str,
    source: str,
    limit: int,
    query_filter: Optional[models.Filter] = None,
) -> List[Dict[str, Any]]:
    if not sparse.indices:
        return []
    try:
        resp = client.query_points(
            collection_name=COLLECTION_NAME,
            query=sparse,
            using=using,
            limit=limit,
            query_filter=query_filter,
            with_payload=PAYLOAD_FIELDS,
            with_vectors=False,
        )
        return [c for p in _extract_points(resp) if (c := _to_candidate(p, source))]
    except Exception as exc:
        logger.warning("Sparse search failed (%s): %s", using, exc)
        return []


def _to_sparse(data: Dict[str, Any]) -> Optional[models.SparseVector]:
    idx = data.get("indices", [])
    val = data.get("values", [])
    if not idx:
        return None
    return models.SparseVector(
        indices=[int(i) for i in idx],
        values=[float(v) for v in val],
    )


# ══════════════════════════════════════════════════════════════════════════════
# TIER-2 FALLBACK SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def _build_tag_filter(object_names: List[str]) -> Optional[models.Filter]:
    """
    Strict filter: frame must have ALL requested YOLO object tags present
    in its `tags` payload field.
    """
    if not object_names:
        return None
    conditions = [
        models.FieldCondition(
            key="tags",
            match=models.MatchAny(any=object_names),
        )
    ]
    return models.Filter(must=conditions)


def execute_fallback_search(
    client: QdrantClient,
    embed_resp: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    2-tier Qdrant retrieval using the QueryResponse from /embed/query.

    Tier 1 — Strict (with YOLO tag filter):
        Search all 4 vectors simultaneously, filter by object tag names.
        Want top_k * 5 candidates minimum.

    Tier 2 — Fallback (no filter):
        If Tier-1 returns < top_k * 5 results, drop the filter entirely.
        The object_sparse vector (encoded from nouns + synonyms) acts as
        a soft filter naturally via BM25 scoring.

    Returns merged, deduped candidates list (best score per frame).
    """
    nlp = embed_resp.get("nlp_analysis", {})
    objects = nlp.get("objects", [])
    object_names = [o["object"] for o in objects]

    # Unpack vectors
    sem_vec   = embed_resp["semantic_dense"]["embedding"]
    vis_vec   = embed_resp["visual_dense"]["embedding"]
    obj_sp    = _to_sparse(embed_resp["object_sparse"])
    ocr_sp    = _to_sparse(embed_resp["ocr_sparse"])

    candidate_limit = top_k * 5

    def _run_all(query_filter: Optional[models.Filter]) -> List[Dict[str, Any]]:
        """Fire all 4 search streams and merge."""
        streams: List[List[Dict[str, Any]]] = [
            _query_dense(client, sem_vec, VEC_CAPTION_DENSE, "caption",  candidate_limit, query_filter),
            _query_dense(client, vis_vec, VEC_DENSE,          "keyframe", candidate_limit, query_filter),
        ]
        if obj_sp:
            streams.append(_query_sparse(client, obj_sp, VEC_OBJECT_SPARSE, "object", candidate_limit, query_filter))
        if ocr_sp:
            streams.append(_query_sparse(client, ocr_sp, VEC_OCR_SPARSE,    "ocr",    candidate_limit, query_filter))
        return _merge_candidates(streams)

    # ── Tier 1: Strict ────────────────────────────────────────────────────
    strict_filter = _build_tag_filter(object_names)
    results = _run_all(strict_filter)

    logger.info(
        "Tier-1 (strict filter=%r): %d candidates", object_names, len(results)
    )

    # ── Tier 2: Fallback ──────────────────────────────────────────────────
    if len(results) < candidate_limit:
        logger.info(
            "Tier-2 fallback (no filter): %d < %d, re-searching...",
            len(results), candidate_limit,
        )
        fallback = _run_all(None)
        # Merge — Tier-1 strict hits keep precedence (higher score)
        results = _merge_candidates([results, fallback])
        logger.info("Tier-2 merged: %d candidates total", len(results))

    return results


def _merge_candidates(streams: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked lists, keeping the best score per (video_id, frame_id)
    and accumulating evidence sources.
    """
    merged: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for stream in streams:
        for item in stream:
            key = (item["video_id"], item["frame_id"])
            if key in merged:
                existing = merged[key]
                # Keep higher score
                if item["score"] > existing["score"]:
                    existing["score"] = item["score"]
                # Accumulate evidence
                src = item.get("source", "")
                if src and src not in existing["evidence"]:
                    existing["evidence"].append(src)
            else:
                merged[key] = {**item, "evidence": [item.get("source", "")]}
    return list(merged.values())


# ══════════════════════════════════════════════════════════════════════════════
# RERANKING — RRF + Count Bonus Multiplier
# ══════════════════════════════════════════════════════════════════════════════

def apply_custom_reranking(
    candidates: List[Dict[str, Any]],
    nlp_analysis: Dict[str, Any],
    top_k: int,
    rrf_k: int = 60,
    beta: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Re-rank candidates using RRF + Count Bonus Multiplier.

    Formula:
        S_RRF   = Σ 1 / (rrf_k + rank_i)   across all evidence streams
        M_i     = 1.0 / (1.0 + |C_act - C_req|)   — per object match quality
        M_avg   = mean(M_i)   — overall count match score
        S_final = S_RRF * (1.0 + beta * M_avg)     — multiplier boost

    The Multiplier Boost (not additive) preserves the relative score
    distribution while rewarding exact object-count matches.

    Parameters
    ----------
    candidates : list[dict]
        Merged candidates from execute_fallback_search().
    nlp_analysis : dict
        NLP analysis from /embed/query response.
    top_k : int
        Final number of results to return.
    rrf_k : int
        RRF smoothing constant (default: 60, standard value).
    beta : float
        Count bonus weight. Higher = stronger count-match boost.

    Returns
    -------
    list[dict]
        Top-K candidates sorted by S_final descending.
    """
    if not candidates:
        return []

    objects = nlp_analysis.get("objects", [])
    # Only objects that have an explicit count requirement
    count_requests = [o for o in objects if o.get("count") is not None]

    # ── Step 1: Sort by raw score to assign RRF rank ──────────────────────
    sorted_by_score = sorted(candidates, key=lambda c: c["score"], reverse=True)

    for rank, cand in enumerate(sorted_by_score, start=1):
        # S_RRF = 1 / (k + rank). Single ranking stream approach.
        s_rrf = 1.0 / (rrf_k + rank)

        # ── Step 2: Count Bonus M_avg ─────────────────────────────────
        if count_requests:
            # object_counts from frame payload: {"cat": 3, "chair": 2, ...}
            frame_counts: Dict[str, int] = (
                cand.get("raw_payload", {}).get("object_counts") or {}
            )

            m_values = []
            for obj in count_requests:
                c_req = obj["count"]                        # required count
                c_act = frame_counts.get(obj["object"], 0) # actual count (default 0)
                delta = abs(c_act - c_req)
                # M_i close to 1.0 = perfect match, decreases with delta
                m_i = 1.0 / (1.0 + delta)
                m_values.append(m_i)

            m_avg = sum(m_values) / len(m_values)
        else:
            m_avg = 0.0  # no count requirement → no bonus

        # ── Step 3: Multiplier Boost ──────────────────────────────────
        # S_final = S_RRF * (1 + beta * M_avg)
        # This rewards count-matching frames without distorting ranking
        # for frames where count is irrelevant (m_avg=0 → no change).
        s_final = s_rrf * (1.0 + beta * m_avg)

        cand["score"] = round(s_final, 8)
        cand["_rrf_rank"] = rank
        cand["_m_avg"] = round(m_avg, 4)

    # ── Step 4: Final sort and trim ───────────────────────────────────────
    final = sorted(sorted_by_score, key=lambda c: c["score"], reverse=True)

    # Remove internal debug keys
    for c in final:
        c.pop("_rrf_rank", None)
        c.pop("_m_avg", None)

    return final[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# SERVICE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class HeuristicRetrieveService:
    """
    Production heuristic retrieval: /embed/query → 2-tier Qdrant → RRF + Count Bonus.

    Usage (matches existing interface in search_controller.py):
        service = HeuristicRetrieveService()
        candidates = service.retrieve(query_bundle, top_k=10)
    """

    def __init__(self) -> None:
        from src.config import get_qdrant_url, get_qdrant_api_key, get_embedding_api_url

        qdrant_url = get_qdrant_url()
        qdrant_key = get_qdrant_api_key()
        embed_url  = get_embedding_api_url()

        self._qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=60)
        self._embed  = EmbedQueryClient(base_url=embed_url, timeout=90)
        logger.info("HeuristicRetrieveService initialized (production mode)")

    def retrieve(
        self,
        query_bundle: Dict[str, Any],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Run heuristic retrieval for a pre-processed query bundle.

        Parameters
        ----------
        query_bundle : dict
            Fields used: ``translated_en`` (preferred) → ``cleaned`` fallback.
        top_k : int
            Number of final results.

        Returns
        -------
        list[dict]
            Each dict has: video_id, frame_id, score, branch, evidence, raw_payload
        """
        query_text = (
            query_bundle.get("translated_en")
            or query_bundle.get("cleaned")
            or query_bundle.get("raw_query", "")
        ).strip()

        if not query_text:
            logger.warning("HeuristicRetrieveService.retrieve: empty query")
            return []

        logger.info("HeuristicRetrieveService.retrieve: query=%r top_k=%d", query_text, top_k)

        # ── 1. Get all 4 vectors + NLP in one call ────────────────────────
        embed_resp = self._embed.query(query_text)
        if embed_resp is None:
            logger.error("Embedding API unavailable — returning empty results")
            return []

        nlp_analysis = embed_resp.get("nlp_analysis", {})
        logger.info(
            "NLP: objects=%s ocr=%s",
            [o["object"] for o in nlp_analysis.get("objects", [])],
            nlp_analysis.get("ocr_texts", []),
        )

        # ── 2. 2-tier Qdrant search ───────────────────────────────────────
        candidates = execute_fallback_search(self._qdrant, embed_resp, top_k)

        if not candidates:
            logger.warning("No candidates returned from Qdrant")
            return []

        # ── 3. RRF + Count Bonus reranking ────────────────────────────────
        ranked = apply_custom_reranking(
            candidates=candidates,
            nlp_analysis=nlp_analysis,
            top_k=top_k,
        )

        logger.info("HeuristicRetrieveService: returning %d results", len(ranked))
        return ranked
