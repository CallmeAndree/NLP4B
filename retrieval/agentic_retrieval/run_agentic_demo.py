from __future__ import annotations
import json

from graph import build_agentic_retrieval_graph
from services.llm_service import LLMService
from services.qdrant_search import QdrantSearchService


class MockLLMService(LLMService):
    def invoke(self, prompt: str) -> str:
        return json.dumps({
            "objects": ["speaker"],
            "attributes": ["red shirt"],
            "actions": ["speaking"],
            "scene": ["outdoor"],
            "text_cues": [],
            "metadata_cues": [],
            "query_type": "visual_event"
        }, ensure_ascii=False)


class MockQdrantSearchService(QdrantSearchService):
    def search_keyframe(self, query_texts, top_k=20):
        return [
            {"video_id": "video_001", "frame_id": 100, "score": 0.71},
            {"video_id": "video_002", "frame_id": 220, "score": 0.65},
        ]

    def search_ocr(self, query_texts, top_k=20):
        return [
            {"video_id": "video_003", "frame_id": 50, "score": 0.20},
        ]

    def search_object(self, query_texts, top_k=20):
        return [
            {"video_id": "video_001", "frame_id": 100, "score": 0.82},
            {"video_id": "video_004", "frame_id": 77, "score": 0.68},
        ]

    def search_metadata(self, query_texts, top_k=20):
        return []

    def search_caption(self, query_texts, top_k=20):
        return [
            {"video_id": "video_001", "frame_id": 100, "score": 0.88},
            {"video_id": "video_005", "frame_id": 310, "score": 0.74},
        ]


def main():
    llm = MockLLMService()
    search_service = MockQdrantSearchService()

    graph = build_agentic_retrieval_graph(llm, search_service)

    initial_state = {
        "raw_query": "Tìm video có một diễn giả mặc áo đỏ phát biểu ngoài trời"
    }

    final_state = graph.invoke(initial_state)

    print("=" * 80)
    print("AGENT TOP-K")
    print("=" * 80)
    for idx, item in enumerate(final_state["agent_topk"], start=1):
        print(
            f"{idx:02d}. video={item['video_id']} | frame={item['frame_id']} "
            f"| agent_score={item['agent_score']:.4f} | evidence={item.get('evidence', [])}"
        )

    print("\n" + "=" * 80)
    print("TRACE LOGS")
    print("=" * 80)
    for log in final_state.get("trace_logs", []):
        print(f"\n[{log['node']}]")
        print(json.dumps(log["payload"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()