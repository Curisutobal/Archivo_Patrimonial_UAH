from typing import Callable, Dict, List, Any

class EventBus:
    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[[Any], None]]] = {}

    def subscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        self._subs.setdefault(event, []).append(handler)

    def publish(self, event: str, payload: Any) -> None:
        for handler in self._subs.get(event, []):
            try:
                handler(payload)
            except Exception:
                # Observers must not break the app
                pass

class LoggingObserver:
    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def __call__(self, payload: Any) -> None:
        try:
            print(f"{self.prefix}{payload}")
        except Exception:
            pass
