from __future__ import annotations

import logging
from typing import Any
import os

from pinecone import Pinecone
from app.models.schemas import FileResult
from app.config import Settings

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.pinecone_api_key or not settings.pinecone_index_name:
            self.pc = None
            self.index = None
            return

        try:
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            self.index_name = settings.pinecone_index_name
            self.index = self.pc.Index(self.index_name)
            logger.info("Connected to Pinecone index: %s", self.index_name)
        except Exception:
            logger.exception("Failed to connect to Pinecone")
            self.pc = None
            self.index = None

    @property
    def configured(self) -> bool:
        return self.index is not None

    def upsert_document(self, file_id: str, content: str, metadata: dict[str, Any]) -> bool:
        """Embed and upsert a document chunk into Pinecone."""
        if not self.configured or not content.strip():
            return False

        try:
            # Chunk the content if it's too long
            chunks = self._chunk_text(content)
            for i, chunk in enumerate(chunks):
                # Use Pinecone Inference for embeddings (llama-text-embed-v2)
                embedding_response = self.pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=[chunk],
                    parameters={"input_type": "passage"}
                )
                vector = embedding_response[0].values
                
                # Metadata must be strings/bools/numbers
                clean_metadata = {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in metadata.items()}
                clean_metadata["text"] = chunk
                clean_metadata["chunk_index"] = i

                self.index.upsert(
                    vectors=[(f"{file_id}_{i}", vector, clean_metadata)]
                )
            return True
        except Exception:
            logger.exception("Failed to upsert to Pinecone for %s", file_id)
            return False

    def query(self, query_text: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search Pinecone for relevant chunks."""
        if not self.configured:
            return []

        try:
            # Embed the query
            embedding_response = self.pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[query_text],
                parameters={"input_type": "query"}
            )
            vector = embedding_response[0].values

            results = self.index.query(
                vector=vector,
                top_k=limit,
                include_metadata=True
            )
            
            output = []
            for match in results.matches:
                meta = match.metadata
                output.append({
                    "content": meta.get("text", ""),
                    "score": match.score,
                    "name": meta.get("name", "Unknown"),
                    "path": meta.get("path", ""),
                    "source": meta.get("source", "local"),
                    "type": meta.get("type", "file")
                })
            return output
        except Exception:
            logger.exception("Pinecone query failed")
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
