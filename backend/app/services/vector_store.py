from __future__ import annotations

import logging
from typing import Any
import json
import requests
import hashlib

from app.config import Settings

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.api_key = settings.pinecone_api_key
        self.index_name = settings.pinecone_index_name
        self.host = None
        
        if not self.api_key or not self.index_name:
            return

        # Discover the index host URL via REST API (Zero SDK dependencies)
        try:
            resp = requests.get(
                f"https://api.pinecone.io/indexes/{self.index_name}",
                headers={"Api-Key": self.api_key, "X-Pinecone-API-Version": "2024-07"},
                timeout=10
            )
            if resp.status_code == 200:
                self.host = f"https://{resp.json()['host']}"
                logger.info("Discovered Pinecone host: %s", self.host)
            else:
                logger.error("Failed to discover Pinecone index: %s", resp.text)
        except Exception:
            logger.exception("Failed to connect to Pinecone Control Plane")

    @property
    def configured(self) -> bool:
        return self.host is not None and self.api_key is not None

    def upsert_document(self, file_id: str, content: str, metadata: dict[str, Any]) -> bool:
        """Embed and upsert via REST API."""
        if not self.configured or not content.strip():
            return False

        try:
            chunks = self._chunk_text(content)
            vectors = []
            
            for i, chunk in enumerate(chunks):
                # Use Pinecone Inference REST API for embeddings
                embed_resp = requests.post(
                    "https://api.pinecone.io/embed",
                    headers={"Api-Key": self.api_key, "Content-Type": "application/json", "X-Pinecone-API-Version": "2024-07"},
                    json={
                        "model": "llama-text-embed-v2",
                        "inputs": [{"text": chunk}],
                        "parameters": {"input_type": "passage", "truncate": "END"}
                    },
                    timeout=20
                )
                if embed_resp.status_code != 200:
                    logger.error("Pinecone embedding failed: %s", embed_resp.text)
                    continue
                
                vector_values = embed_resp.json()["data"][0]["values"]
                
                # Metadata must be simple types
                clean_metadata = {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in metadata.items()}
                clean_metadata["text"] = chunk
                clean_metadata["chunk_index"] = i

                vectors.append({
                    "id": f"{file_id}_{i}",
                    "values": vector_values,
                    "metadata": clean_metadata
                })

            # Batch upsert to the data plane
            if vectors:
                upsert_resp = requests.post(
                    f"{self.host}/vectors/upsert",
                    headers={"Api-Key": self.api_key, "Content-Type": "application/json"},
                    json={"vectors": vectors},
                    timeout=20
                )
                return upsert_resp.status_code == 200
            return False
        except Exception:
            logger.exception("REST Upsert failed")
            return False

    def query(self, query_text: str, limit: int = 5) -> list[dict[str, Any]]:
        """Query via REST API."""
        if not self.configured:
            return []

        try:
            # Embed the query
            embed_resp = requests.post(
                "https://api.pinecone.io/embed",
                headers={"Api-Key": self.api_key, "Content-Type": "application/json", "X-Pinecone-API-Version": "2024-07"},
                json={
                    "model": "llama-text-embed-v2",
                    "inputs": [{"text": query_text}],
                    "parameters": {"input_type": "query", "truncate": "END"}
                },
                timeout=20
            )
            if embed_resp.status_code != 200:
                return []
            
            vector_values = embed_resp.json()["data"][0]["values"]

            # Query the data plane
            query_resp = requests.post(
                f"{self.host}/query",
                headers={"Api-Key": self.api_key, "Content-Type": "application/json"},
                json={
                    "vector": vector_values,
                    "topK": limit,
                    "includeMetadata": True
                },
                timeout=20
            )
            
            if query_resp.status_code != 200:
                return []
            
            matches = query_resp.json().get("matches", [])
            output = []
            for match in matches:
                meta = match.get("metadata", {})
                output.append({
                    "content": meta.get("text", ""),
                    "score": match.get("score", 0),
                    "name": meta.get("name", "Unknown"),
                    "path": meta.get("path", ""),
                    "source": meta.get("source", "local"),
                    "type": meta.get("type", "file")
                })
            return output
        except Exception:
            logger.exception("REST Query failed")
            return []

    def _chunk_text(self, text: str, chunk_size: int = 1500) -> list[str]:
        """Simple character-based chunking."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
