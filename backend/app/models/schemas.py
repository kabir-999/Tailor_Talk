from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    drive = "drive"
    local = "local"
    hybrid = "hybrid"


class FileResult(BaseModel):
    name: str
    type: str = "unknown"
    modified_time: str | None = None
    link: str | None = None
    file_id: str | None = None
    path: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None
    search_mode: SearchMode = SearchMode.local


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    results: list[FileResult] = Field(default_factory=list)
    query_explanation: str | None = None
    suggested_followups: list[str] = Field(default_factory=list)
    search_history: list[dict[str, Any]] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    search_mode: SearchMode = SearchMode.local
    limit: int = Field(default=10, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    results: list[FileResult]
    query_explanation: str | None = None
    generated_drive_q: str | None = None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    groq_configured: bool
    drive_configured: bool
    local_folder_exists: bool
    local_folder: str
    search_uploads_only: bool = True
    uploaded_file_count: int = 0


class UploadResponse(BaseModel):
    filename: str
    saved_path: str
    size: int
    message: str


class UploadListResponse(BaseModel):
    files: list[FileResult] = Field(default_factory=list)


class DeleteUploadResponse(BaseModel):
    deleted_path: str
    message: str


class DriveConnectionResponse(BaseModel):
    connected: bool
    folder_id: str | None = None
    file_count: int = 0
    sample_files: list[FileResult] = Field(default_factory=list)
    generated_drive_q: str | None = None
    message: str
