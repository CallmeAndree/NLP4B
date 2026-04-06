from __future__ import annotations
from state import AgentState
from services.scoring import normalize_weights


def compute_modality_weights(intent: dict) -> dict[str, float]:
    weights = {
        "keyframe": 0.20,
        "ocr": 0.10,
        "object": 0.20,
        "metadata": 0.10,
        "caption": 0.40,
    }

    text_cues = intent.get("text_cues", []) or []
    metadata_cues = intent.get("metadata_cues", []) or []
    objects = intent.get("objects", []) or []
    actions = intent.get("actions", []) or []
    scene = intent.get("scene", []) or []
    query_type = intent.get("query_type", "mixed")

    if text_cues:
        weights["ocr"] += 0.25

    if metadata_cues:
        weights["metadata"] += 0.20

    if len(objects) >= 1:
        weights["object"] += 0.15

    if actions or scene:
        weights["caption"] += 0.15

    if query_type == "text_in_image":
        weights["ocr"] += 0.25
        weights["caption"] -= 0.10

    if query_type == "metadata_hint":
        weights["metadata"] += 0.25

    if query_type == "visual_object":
        weights["object"] += 0.10
        weights["keyframe"] += 0.05

    if query_type == "visual_event":
        weights["caption"] += 0.10
        weights["keyframe"] += 0.05

    return normalize_weights(weights)


def modality_routing_node(state: AgentState) -> AgentState:
    intent = state["query_intent"]
    state["routing_weights"] = compute_modality_weights(intent)

    state.setdefault("trace_logs", []).append({
        "node": "modality_routing",
        "payload": {"routing_weights": state["routing_weights"]},
    })
    return state