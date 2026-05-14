from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class SQLiteMemory:
    def __init__(self, db_path: str | Path, max_messages: int = 12) -> None:
        self.db_path = Path(db_path)
        self.max_messages = max_messages
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL,
                        type TEXT NOT NULL,
                        content TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL,
                        entry_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_data (
                        conversation_id TEXT PRIMARY KEY,
                        last_results_json TEXT,
                        last_extracted_text TEXT
                    )
                    """
                )
                conn.commit()

    def new_id(self) -> str:
        return str(uuid4())

    def get_messages(self, conversation_id: str) -> list[BaseMessage]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT type, content FROM (SELECT * FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                    (conversation_id, self.max_messages),
                )
                rows = cursor.fetchall()
                
        result: list[BaseMessage] = []
        for type_, content in rows:
            if type_ == "human":
                result.append(HumanMessage(content=content))
            elif type_ == "ai":
                result.append(AIMessage(content=content))
        return result

    def append_user(self, conversation_id: str, content: str) -> None:
        self._append(conversation_id, "human", content)

    def append_ai(self, conversation_id: str, content: str) -> None:
        self._append(conversation_id, "ai", content)

    def _append(self, conversation_id: str, type_: str, content: str) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO messages (conversation_id, type, content) VALUES (?, ?, ?)",
                    (conversation_id, type_, content),
                )
                conn.commit()

    def add_search(self, conversation_id: str, entry: dict[str, Any]) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO history (conversation_id, entry_json) VALUES (?, ?)",
                    (conversation_id, json.dumps(entry)),
                )
                conn.commit()

    def search_history(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT entry_json FROM (SELECT * FROM history WHERE conversation_id = ? ORDER BY id DESC LIMIT 10) ORDER BY id ASC",
                    (conversation_id,),
                )
                rows = cursor.fetchall()
        return [json.loads(row[0]) for row in rows]

    def set_last_results(self, conversation_id: str, results: list[dict[str, Any]]) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO session_data (conversation_id, last_results_json) VALUES (?, ?) "
                    "ON CONFLICT(conversation_id) DO UPDATE SET last_results_json=excluded.last_results_json",
                    (conversation_id, json.dumps(results[:10])),
                )
                conn.commit()

    def last_results(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT last_results_json FROM session_data WHERE conversation_id = ?",
                    (conversation_id,),
                )
                row = cursor.fetchone()
        
        if row and row[0]:
            return json.loads(row[0])
        return []

    def set_last_extracted_text(self, conversation_id: str, text: str) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO session_data (conversation_id, last_extracted_text) VALUES (?, ?) "
                    "ON CONFLICT(conversation_id) DO UPDATE SET last_extracted_text=excluded.last_extracted_text",
                    (conversation_id, text),
                )
                conn.commit()

    def last_extracted_text(self, conversation_id: str) -> str:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT last_extracted_text FROM session_data WHERE conversation_id = ?",
                    (conversation_id,),
                )
                row = cursor.fetchone()
        return row[0] if row and row[0] else ""

    def reset(self, conversation_id: str) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                conn.execute("DELETE FROM history WHERE conversation_id = ?", (conversation_id,))
                conn.execute("DELETE FROM session_data WHERE conversation_id = ?", (conversation_id,))
                conn.commit()
