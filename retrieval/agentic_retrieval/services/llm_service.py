from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LLMService:
    model_name: str = "gpt-4.1-mini"

    def invoke(self, prompt: str) -> str:
        """
        Hàm stub.
        Sau này thay bằng:
        - ChatOpenAI.invoke(...)
        - hoặc client.responses.create(...)
        """
        raise NotImplementedError("Connect your actual LLM here.")