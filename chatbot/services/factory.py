import os
from typing import Any, Dict, Optional

from .llm_proxy import GeminiClientProxy

class ServiceFactory:
    """Abstract Factory for creating app services based on environment."""

    def __init__(self, genai: Any, genai_available: bool) -> None:
        self._genai = genai
        self._genai_available = genai_available
        self._gemini = GeminiClientProxy(genai, genai_available)

    def make_embedding(self):
        """Return a callable(text)->Optional[list[float]] using Gemini when available."""
        def _embed(text: str) -> Optional[list]:
            if not self._genai_available:
                return None
            result = self._gemini.embed(text, model="models/text-embedding-004", task_type="retrieval_document")
            if not result:
                return None
            return result.get("embedding")
        return _embed

    def make_query_embedding(self):
        def _embed_query(text: str) -> Optional[list]:
            if not self._genai_available:
                return None
            result = self._gemini.embed(text, model="models/text-embedding-004", task_type="retrieval_query")
            if not result:
                return None
            return result.get("embedding")
        return _embed_query

    def make_responder(self):
        def _respond(prompt: str) -> Optional[str]:
            if not self._genai_available:
                return None
            return self._gemini.generate(prompt, model_name="gemini-2.0-flash-exp")
        return _respond
