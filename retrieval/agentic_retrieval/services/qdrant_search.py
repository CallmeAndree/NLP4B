from __future__ import annotations
from typing import Any, Dict, List


class QdrantSearchService:
    def __init__(self) -> None:
        # inject qdrant client thật ở đây sau
        pass

    def search_keyframe(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        return []

    def search_ocr(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        return []

    def search_object(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        return []

    def search_metadata(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        return []

    def search_caption(self, query_texts: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        return []