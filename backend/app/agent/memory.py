from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ConversationMemory:
    def __init__(self, max_messages: int = 12) -> None:
        self.max_messages = max_messages
        self._messages: dict[str, list[BaseMessage]] = defaultdict(list)
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._last_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = Lock()

    def new_id(self) -> str:
        return str(uuid4())

    def get_messages(self, conversation_id: str) -> list[BaseMessage]:
        with self._lock:
            return list(self._messages[conversation_id][-self.max_messages :])

    def append_user(self, conversation_id: str, content: str) -> None:
        self._append(conversation_id, HumanMessage(content=content))

    def append_ai(self, conversation_id: str, content: str) -> None:
        self._append(conversation_id, AIMessage(content=content))

    def add_search(self, conversation_id: str, entry: dict[str, Any]) -> None:
        with self._lock:
            self._history[conversation_id].append(entry)
            self._history[conversation_id] = self._history[conversation_id][-10:]

    def search_history(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history[conversation_id])

    def set_last_results(self, conversation_id: str, results: list[dict[str, Any]]) -> None:
        with self._lock:
            self._last_results[conversation_id] = results[:10]

    def last_results(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._last_results[conversation_id])

    def reset(self, conversation_id: str) -> None:
        with self._lock:
            self._messages.pop(conversation_id, None)
            self._history.pop(conversation_id, None)
            self._last_results.pop(conversation_id, None)

    def _append(self, conversation_id: str, message: BaseMessage) -> None:
        with self._lock:
            self._messages[conversation_id].append(message)
            self._messages[conversation_id] = self._messages[conversation_id][-self.max_messages :]
