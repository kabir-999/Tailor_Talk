from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.agent.chains import DriveQueryChain
from app.models.schemas import FileResult
from app.services.drive_service import GoogleDriveService
from app.services.local_search import LocalSearchService


class DriveSearchInput(BaseModel):
    query: str = Field(description="The user's natural-language search request.")
    drive_q: str | None = Field(default=None, description="Valid Google Drive API v3 q query generated from the request.")
    limit: int = Field(default=10, ge=1, le=50)


class LocalSearchInput(BaseModel):
    query: str = Field(description="The user's natural-language search request.")
    limit: int = Field(default=10, ge=1, le=50)


class FileSummaryInput(BaseModel):
    path: str = Field(description="Relative or absolute local Assignment folder path to summarize.")


class DriveFileSummaryInput(BaseModel):
    file_id: str = Field(description="Google Drive file ID to download and extract content from.")
    mime_type: str = Field(description="MIME type of the file (e.g. application/pdf, image/png).")
    filename: str = Field(description="Display name of the file.")


def serialize_results(results: list[FileResult], extra: dict[str, Any] | None = None) -> str:
    payload = extra or {}
    payload["results"] = [result.model_dump() for result in results]
    return json.dumps(payload, default=str)


def build_tools(
    drive_service: GoogleDriveService,
    local_service: LocalSearchService,
    drive_query_chain: DriveQueryChain,
) -> list[StructuredTool]:
    def drive_search(query: str, drive_q: str | None = None, limit: int = 10) -> str:
        if not drive_q:
            plan = drive_query_chain.generate(query)
            drive_q = plan.q
            generated_explanation = plan.explanation
        else:
            generated_explanation = "LLM supplied a Drive API q query."
        results, explanation, generated_q = drive_service.search(query=query, drive_q=drive_q, limit=limit)
        return serialize_results(
            results,
            {
                "tool": "DriveSearchTool",
                "query_explanation": f"{generated_explanation} {explanation}".strip(),
                "generated_drive_q": generated_q,
            },
        )

    def local_search(query: str, limit: int = 10) -> str:
        results, explanation = local_service.search(query=query, limit=limit)
        return serialize_results(results, {"tool": "LocalSearchTool", "query_explanation": explanation})

    def summarize_file(path: str) -> str:
        summary = local_service.summarize_file(path)
        return json.dumps({"tool": "FileSummaryTool", "path": path, "summary": summary})

    def drive_summarize_file(file_id: str, mime_type: str, filename: str) -> str:
        """Download a file from Google Drive and extract its text content."""
        try:
            content = drive_service.extract_content(
                file_id=file_id, mime_type=mime_type, filename=filename, max_chars=3000
            )
            return json.dumps({
                "tool": "DriveFileSummaryTool",
                "file_id": file_id,
                "filename": filename,
                "summary": content,
            })
        except Exception as exc:
            return json.dumps({
                "tool": "DriveFileSummaryTool",
                "file_id": file_id,
                "filename": filename,
                "error": str(exc),
                "summary": "",
            })

    return [
        StructuredTool.from_function(
            name="DriveSearchTool",
            description=(
                "Search Google Drive. Always pass a valid Google Drive API v3 q string in drive_q when possible. "
                "Use for Google Drive mode and Drive-specific file discovery."
            ),
            func=drive_search,
            args_schema=DriveSearchInput,
        ),
        StructuredTool.from_function(
            name="LocalSearchTool",
            description="Search the local Assignment folder by filename, extracted document text, and semantic similarity.",
            func=local_search,
            args_schema=LocalSearchInput,
        ),
        StructuredTool.from_function(
            name="FileSummaryTool",
            description="Summarize or extract text from a specific local file returned by LocalSearchTool.",
            func=summarize_file,
            args_schema=FileSummaryInput,
        ),
        StructuredTool.from_function(
            name="DriveFileSummaryTool",
            description=(
                "Download a Google Drive file and extract its readable text content. "
                "Use when the user asks to read, OCR, summarize, or extract content from a Drive file. "
                "Requires the file_id, mime_type, and filename from a previous DriveSearchTool result."
            ),
            func=drive_summarize_file,
            args_schema=DriveFileSummaryInput,
        ),
    ]
