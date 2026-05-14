from __future__ import annotations

from datetime import datetime, timezone
import logging

from fastapi import File, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.agent.graph import DriveDiscoveryAgent
from app.agent.memory import ConversationMemory
from app.config import get_settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteUploadResponse,
    DriveConnectionResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    UploadListResponse,
    UploadResponse,
)
from app.services.drive_service import GoogleDriveService
from app.services.local_search import LocalSearchService
from app.utils.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Conversational Google Drive File Discovery Agent",
    version="1.0.0",
    description="FastAPI backend for AI-powered Google Drive and local Assignment folder discovery.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

drive_service = GoogleDriveService(settings)
local_service = LocalSearchService(settings.local_assignment_dir, uploads_only=settings.search_uploads_only)
memory = ConversationMemory()
agent = DriveDiscoveryAgent(settings, drive_service, local_service, memory)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": "Conversational Google Drive File Discovery Agent",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc),
        groq_configured=bool(settings.groq_api_key),
        drive_configured=drive_service.configured,
        local_folder_exists=local_service.configured,
        local_folder=str(settings.local_assignment_dir),
        search_uploads_only=settings.search_uploads_only,
        uploaded_file_count=len(local_service.list_uploads()) if local_service.configured else 0,
    )


@app.get("/drive/test", response_model=DriveConnectionResponse)
async def test_drive_connection() -> DriveConnectionResponse:
    if not settings.google_drive_folder_id:
        raise HTTPException(status_code=400, detail="GOOGLE_DRIVE_FOLDER_ID is not set.")
    try:
        results, _explanation, generated_q = drive_service.search(query="", drive_q="trashed=false", limit=5)
        return DriveConnectionResponse(
            connected=True,
            folder_id=settings.google_drive_folder_id,
            file_count=len(results),
            sample_files=results,
            generated_drive_q=generated_q,
            message="Successfully connected to the configured Google Drive folder.",
        )
    except Exception as exc:
        logger.exception("Drive connection test failed")
        return DriveConnectionResponse(
            connected=False,
            folder_id=settings.google_drive_folder_id,
            generated_drive_q=None,
            message=str(exc),
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await agent.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            search_mode=request.search_mode,
        )
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    try:
        return await agent.search(
            query=request.query,
            search_mode=request.search_mode,
            limit=request.limit,
        )
    except Exception as exc:
        logger.exception("Search request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        saved_path = local_service.save_upload(file.filename or "uploaded-file", content)
        relative_path = saved_path.relative_to(local_service.root)
        return UploadResponse(
            filename=saved_path.name,
            saved_path=str(relative_path),
            size=len(content),
            message="File uploaded and added to local search.",
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/uploads", response_model=UploadListResponse)
async def list_uploads() -> UploadListResponse:
    try:
        return UploadListResponse(files=local_service.list_uploads())
    except Exception as exc:
        logger.exception("Listing uploads failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/uploads/{upload_path:path}", response_model=DeleteUploadResponse)
async def delete_upload(upload_path: str) -> DeleteUploadResponse:
    try:
        deleted = local_service.delete_upload(upload_path)
        return DeleteUploadResponse(
            deleted_path=str(deleted.relative_to(local_service.root)),
            message="Uploaded file removed from the searchable corpus.",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Deleting upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
