from typing import Any, Dict, Optional

class GeminiClientProxy:
    """Proxy around google.generativeai client providing simple guards."""

    def __init__(self, genai: Any, api_key_present: bool) -> None:
        self._genai = genai
        self._enabled = api_key_present

    def embed(self, text: str, *, model: str, task_type: str) -> Optional[Dict[str, Any]]:
        if not self._enabled:
            return None
        try:
            result = self._genai.embed_content(
                model=model,
                content=text,
                task_type=task_type,
            )
            return {"embedding": result.get("embedding")}
        except Exception:
            # Disable on repeated failures could be added here
            return None

    def generate(self, prompt: str, *, model_name: str) -> Optional[str]:
        if not self._enabled:
            return None
        try:
            model = self._genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return getattr(response, "text", None)
        except Exception:
            return None
