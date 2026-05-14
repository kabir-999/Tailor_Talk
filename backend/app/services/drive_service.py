from __future__ import annotations

import io
import logging
import json
import base64
import re
import tempfile
from calendar import month_name
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.config import Settings
from app.models.schemas import FileResult

logger = logging.getLogger(__name__)

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

MIME_ALIASES = {
    "pdf": "application/pdf",
    "pdfs": "application/pdf",
    "spreadsheet": "application/vnd.google-apps.spreadsheet",
    "spreadsheets": "application/vnd.google-apps.spreadsheet",
    "sheet": "application/vnd.google-apps.spreadsheet",
    "sheets": "application/vnd.google-apps.spreadsheet",
    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "document": "application/vnd.google-apps.document",
    "documents": "application/vnd.google-apps.document",
    "doc": "application/vnd.google-apps.document",
    "docs": "application/vnd.google-apps.document",
    "image": "image/",
    "images": "image/",
    "photo": "image/",
    "photos": "image/",
    "presentation": "application/vnd.google-apps.presentation",
    "slides": "application/vnd.google-apps.presentation",
}


def escape_drive_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


class DriveQueryBuilder:
    """Conservative fallback query builder for natural language Drive searches."""

    def __init__(self, folder_id: str | None = None) -> None:
        self.folder_id = folder_id

    def build(self, query: str) -> tuple[str, str]:
        lowered = query.lower()
        clauses: list[str] = ["trashed=false"]
        explanation: list[str] = []

        if self.folder_id:
            clauses.append(f"'{escape_drive_literal(self.folder_id)}' in parents")
            explanation.append("limited to the configured Drive folder")

        for label, mime in MIME_ALIASES.items():
            if label in lowered:
                if mime.endswith("/"):
                    clauses.append(f"mimeType contains '{mime}'")
                else:
                    clauses.append(f"mimeType='{mime}'")
                explanation.append(f"filtered to {label} files")
                break

        now = datetime.now(timezone.utc)
        if "last week" in lowered:
            start = now - timedelta(days=7)
            clauses.append(f"modifiedTime > '{start.strftime('%Y-%m-%dT00:00:00')}'")
            explanation.append("modified in the last 7 days")
        elif "this week" in lowered:
            start = now - timedelta(days=now.weekday())
            clauses.append(f"modifiedTime > '{start.strftime('%Y-%m-%dT00:00:00')}'")
            explanation.append("modified this week")
        elif "today" in lowered:
            clauses.append(f"modifiedTime > '{now.strftime('%Y-%m-%dT00:00:00')}'")
            explanation.append("modified today")

        for month_number, name in enumerate(month_name):
            if month_number and name.lower() in lowered:
                year = now.year
                next_month = month_number + 1
                next_year = year
                if next_month == 13:
                    next_month = 1
                    next_year += 1
                date_field = "createdTime" if any(word in lowered for word in ["uploaded", "created", "added"]) else "modifiedTime"
                clauses.append(f"{date_field} >= '{year}-{month_number:02d}-01T00:00:00'")
                clauses.append(f"{date_field} < '{next_year}-{next_month:02d}-01T00:00:00'")
                explanation.append(f"{date_field} filtered to {name} {year}")
                break

        terms = self._extract_terms(query)
        exact_match = re.search(r"['\"]([^'\"]+\.[A-Za-z0-9]{2,5})['\"]", query)
        if exact_match or ("exact" in lowered and terms):
            filename = escape_drive_literal(exact_match.group(1) if exact_match else " ".join(terms))
            clauses.append(f"name='{filename}'")
            explanation.append(f"exact filename match for '{filename}'")
        elif terms:
            term = escape_drive_literal(" ".join(terms[:4]))
            if any(word in lowered for word in ["containing", "contains", "content", "inside"]):
                clauses.append(f"fullText contains '{term}'")
                explanation.append(f"full-text search for '{term}'")
            else:
                clauses.append(f"(name contains '{term}' or fullText contains '{term}')")
                explanation.append(f"name or content search for '{term}'")

        return " and ".join(clauses), "; ".join(explanation) or "general Drive search"

    @staticmethod
    def _extract_terms(query: str) -> list[str]:
        stopwords = {
            "find",
            "show",
            "search",
            "files",
            "file",
            "about",
            "related",
            "uploaded",
            "modified",
            "this",
            "last",
            "week",
            "today",
            "documents",
            "document",
            "spreadsheets",
            "spreadsheet",
            "images",
            "image",
            "pdf",
            "pdfs",
            "containing",
            "contains",
        }
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in query)
        return [token for token in cleaned.split() if token.lower() not in stopwords]


class GoogleDriveService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._service: Any | None = None
        self.query_builder = DriveQueryBuilder(settings.google_drive_folder_id)

    @property
    def configured(self) -> bool:
        credentials_path = self.settings.google_application_credentials
        return bool(
            self.settings.google_service_account_json
            or self.settings.google_service_account_json_b64
            or (credentials_path and Path(credentials_path).expanduser().exists())
        )

    def _client(self) -> Any:
        if self._service is not None:
            return self._service
        if not self.configured:
            raise RuntimeError(
                "Google Drive is not configured. Set GOOGLE_SERVICE_ACCOUNT_JSON, "
                "GOOGLE_SERVICE_ACCOUNT_JSON_B64, or GOOGLE_APPLICATION_CREDENTIALS."
            )
        if self.settings.google_service_account_json or self.settings.google_service_account_json_b64:
            credentials_json = self.settings.google_service_account_json
            if self.settings.google_service_account_json_b64:
                credentials_json = base64.b64decode(self.settings.google_service_account_json_b64).decode("utf-8")
            credentials_info = json.loads(credentials_json or "{}")
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=DRIVE_SCOPES,
            )
        else:
            credentials = service_account.Credentials.from_service_account_file(
                str(Path(self.settings.google_application_credentials or "").expanduser()),
                scopes=DRIVE_SCOPES,
            )
        self._service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        return self._service

    def search(self, query: str, drive_q: str | None = None, limit: int = 10) -> tuple[list[FileResult], str, str]:
        q, explanation = (drive_q, "LLM-generated Google Drive query") if drive_q else self.query_builder.build(query)
        if self.settings.google_drive_folder_id and self.settings.google_drive_folder_id not in q:
            q = f"'{escape_drive_literal(self.settings.google_drive_folder_id)}' in parents and trashed=false and ({q})"

        logger.info("Drive search q=%s", q)
        files: list[dict[str, Any]] = []
        page_token: str | None = None
        try:
            while len(files) < limit:
                response = (
                    self._client()
                    .files()
                    .list(
                        q=q,
                        spaces="drive",
                        fields="nextPageToken, files(id,name,mimeType,modifiedTime,webViewLink,size)",
                        pageSize=min(100, max(limit - len(files), 1)),
                        pageToken=page_token,
                    )
                    .execute()
                )
                files.extend(response.get("files", []))
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        except HttpError as exc:
            logger.exception("Google Drive API search failed")
            raise RuntimeError(f"Google Drive search failed: {exc.reason}") from exc

        results = [self._to_result(file, query, index) for index, file in enumerate(files[:limit])]
        return results, explanation, q

    def _to_result(self, file: dict[str, Any], query: str, index: int) -> FileResult:
        name = file.get("name", "")
        confidence = self._score(name, file.get("mimeType", ""), query, index)
        return FileResult(
            name=name,
            type=file.get("mimeType", "unknown"),
            modified_time=file.get("modifiedTime"),
            link=file.get("webViewLink"),
            file_id=file.get("id"),
            confidence=confidence,
            source="google_drive",
            summary=f"{name} matched the Drive search criteria.",
            metadata={"size": file.get("size")},
        )

    @staticmethod
    def _score(name: str, mime_type: str, query: str, index: int) -> float:
        query_terms = {term.lower() for term in DriveQueryBuilder._extract_terms(query)}
        name_terms = set(name.lower().replace(".", " ").split())
        overlap = len(query_terms & name_terms) / max(len(query_terms), 1)
        type_bonus = 0.15 if any(alias in query.lower() and mime in mime_type for alias, mime in MIME_ALIASES.items()) else 0.0
        rank_penalty = min(index * 0.03, 0.25)
        return round(max(0.35, min(0.98, 0.65 + overlap * 0.25 + type_bonus - rank_penalty)), 2)

    # ── Google-native export MIME mapping ──────────────────────────────────
    _EXPORT_MAP: dict[str, tuple[str, str]] = {
        "application/vnd.google-apps.document": ("text/plain", ".txt"),
        "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
        "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
        "application/vnd.google-apps.drawing": ("application/pdf", ".pdf"),
    }

    def download_file(self, file_id: str, mime_type: str) -> tuple[bytes, str]:
        """Download or export a file from Drive.

        Returns (raw_bytes, effective_extension).
        For Google-native types the file is exported; for regular files it is
        downloaded directly.
        """
        export_info = self._EXPORT_MAP.get(mime_type)
        if export_info:
            export_mime, ext = export_info
            data = self._client().files().export(fileId=file_id, mimeType=export_mime).execute()
            return data, ext

        # Regular binary file – download via media
        request = self._client().files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        from googleapiclient.http import MediaIoBaseDownload

        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        ext = Path(mime_type.split("/")[-1]).suffix or ""
        return buffer.getvalue(), ext

    def extract_content(self, file_id: str, mime_type: str, filename: str, max_chars: int = 3000) -> str:
        """Download a Drive file and extract readable text from it.

        Uses a temporary file and delegates to LocalSearchService's extraction
        helpers so every supported format is handled identically to local files.
        """
        from app.services.local_search import LocalSearchService

        raw_bytes, export_ext = self.download_file(file_id, mime_type)

        # Determine a useful suffix for the temp file
        suffix = Path(filename).suffix.lower()
        if not suffix or suffix == ".":
            suffix = export_ext or ".bin"

        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="drive_extract_")
            tmp_path = Path(tmp_dir) / f"drive_file{suffix}"
            tmp_path.write_bytes(raw_bytes if isinstance(raw_bytes, bytes) else raw_bytes.encode("utf-8"))

            # Re-use the local-search extraction pipeline
            service = LocalSearchService.__new__(LocalSearchService)
            service.root = Path(tmp_dir)
            service.uploads_only = False
            service.cache_dir = Path(tmp_dir) / ".cache"
            doc = service._load_document(tmp_path, allow_ocr=True)
            text = " ".join(doc.content.split())
            return text[:max_chars] + ("..." if len(text) > max_chars else "")
        except Exception as exc:
            logger.exception("Drive content extraction failed for %s", filename)
            return f"Failed to extract content from {filename}: {exc}"
        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

