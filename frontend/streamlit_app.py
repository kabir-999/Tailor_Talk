from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import html
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv()

st.set_page_config(
    page_title="Local File Discovery Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 1080px; padding-top: 1.6rem; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(120,120,120,.18); }
      .result-card {
        border: 1px solid rgba(120,120,120,.22);
        border-radius: 8px;
        padding: 12px 14px;
        margin: 8px 0;
        background: rgba(250,250,250,.03);
      }
      .muted { color: rgba(120,120,120,.95); font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("search_mode", "local")
    st.session_state.setdefault("last_results", [])


class LocalResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(self.payload.get("detail", "Request failed"))


@st.cache_resource
def direct_services() -> dict[str, Any]:
    import sys
    # Force clear stale memory module from sys.modules if it exists
    sys.modules.pop("app.agent.memory", None)
    
    from app.agent.graph import DriveDiscoveryAgent
    from app.agent.sqlite_memory import SQLiteMemory
    from app.config import get_settings
    from app.services.drive_service import GoogleDriveService
    from app.services.local_search import LocalSearchService
    from app.utils.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    drive_service = GoogleDriveService(settings)
    local_service = LocalSearchService(
        settings.local_assignment_dir,
        uploads_only=settings.search_uploads_only,
    )
    memory = SQLiteMemory(settings.local_assignment_dir / "memory.db")
    agent = DriveDiscoveryAgent(settings, drive_service, local_service, memory)
    return {
        "settings": settings,
        "drive_service": drive_service,
        "local_service": local_service,
        "agent": agent,
    }


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def backend_get(path: str) -> requests.Response:
    services = direct_services()
    settings = services["settings"]
    drive_service = services["drive_service"]
    local_service = services["local_service"]
    if path == "/health":
        return LocalResponse(
            {
                "status": "ok",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "gemini_configured": bool(settings.gemini_api_key),
                "drive_configured": drive_service.configured,
                "local_folder_exists": local_service.configured,
                "local_folder": str(settings.local_assignment_dir),
                "search_uploads_only": settings.search_uploads_only,
                "uploaded_file_count": len(local_service.list_uploads()) if local_service.configured else 0,
            }
        )
    if path == "/uploads":
        return LocalResponse({"files": [item.model_dump() for item in local_service.list_uploads()]})
    return LocalResponse({"detail": "Not found"}, status_code=404)


def backend_post(path: str, payload: dict[str, Any]) -> requests.Response:
    agent = direct_services()["agent"]
    from app.models.schemas import SearchMode

    search_mode = SearchMode(payload.get("search_mode", "local"))
    if path == "/chat":
        response = run_async(
            agent.chat(
                message=payload["message"],
                conversation_id=payload.get("conversation_id"),
                search_mode=search_mode,
            )
        )
        return LocalResponse(response.model_dump())
    if path == "/search":
        response = run_async(
            agent.search(
                query=payload["query"],
                search_mode=search_mode,
                limit=payload.get("limit", 10),
            )
        )
        return LocalResponse(response.model_dump())
    return LocalResponse({"detail": "Not found"}, status_code=404)


def backend_upload(file: Any) -> requests.Response:
    local_service = direct_services()["local_service"]
    content = file.getvalue()
    if not content:
        return LocalResponse({"detail": "Uploaded file is empty."}, status_code=400)
    try:
        saved_path = local_service.save_upload(file.name or "uploaded-file", content)
        relative_path = saved_path.relative_to(local_service.root)
        return LocalResponse(
            {
                "filename": saved_path.name,
                "saved_path": str(relative_path),
                "size": len(content),
                "message": "File uploaded and added to local search.",
            }
        )
    except ValueError as exc:
        return LocalResponse({"detail": str(exc)}, status_code=400)


def backend_delete_upload(path: str) -> requests.Response:
    local_service = direct_services()["local_service"]
    try:
        deleted = local_service.delete_upload(path)
        return LocalResponse(
            {
                "deleted_path": str(deleted.relative_to(local_service.root)),
                "message": "Uploaded file removed from the searchable corpus.",
            }
        )
    except FileNotFoundError as exc:
        return LocalResponse({"detail": str(exc)}, status_code=404)
    except ValueError as exc:
        return LocalResponse({"detail": str(exc)}, status_code=400)


def uploaded_files() -> list[dict[str, Any]]:
    try:
        response = backend_get("/uploads")
        response.raise_for_status()
        return response.json().get("files", [])
    except Exception:
        return []


def render_result(result: dict[str, Any]) -> None:
    link = result.get("link")
    path = result.get("path")
    if link:
        locator = f'<a href="{html.escape(link)}" target="_blank" rel="noopener">Open file</a>'
    elif path:
        locator = f"<code>{html.escape(path)}</code>"
    else:
        locator = "<code>No link available</code>"
    st.markdown(
        f"""
        <div class="result-card">
          <strong>{html.escape(result.get("name", "Untitled"))}</strong>
          <div class="muted">
            Source: {html.escape(result.get("source", "unknown"))} · Type: {html.escape(result.get("type", "unknown"))} ·
            Modified: {html.escape(result.get("modified_time") or "unknown")} ·
            Confidence: {float(result.get("confidence", 0)):.2f}
          </div>
          <div>{locator}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


init_state()

with st.sidebar:
    st.header("Controls")
    st.session_state.search_mode = "hybrid"

    st.subheader("Upload files")
    uploads = st.file_uploader(
        "Add files to the searchable corpus",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt", "png", "jpg", "jpeg", "webp", "gif", "mp4", "mov", "m4v"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Upload selected files", use_container_width=True):
        with st.spinner("Uploading..."):
            try:
                uploaded_names = []
                for upload in uploads:
                    upload_response = backend_upload(upload)
                    upload_response.raise_for_status()
                    uploaded_names.append(upload_response.json()["filename"])
                st.success(f"Uploaded {len(uploaded_names)} file(s): {', '.join(uploaded_names)}")
                st.rerun()
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    current_uploads = uploaded_files()
    if current_uploads:
        with st.expander("Uploaded searchable files", expanded=True):
            for item in current_uploads:
                left, right = st.columns([0.78, 0.22], vertical_alignment="center")
                with left:
                    st.caption(f"{item['name']} · {item['type']}")
                    st.caption(item.get("path", ""))
                with right:
                    if st.button("Remove", key=f"remove-{item.get('path')}", use_container_width=True):
                        try:
                            response = backend_delete_upload(item.get("path", ""))
                            response.raise_for_status()
                            st.success(f"Removed {item['name']}")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Remove failed: {exc}")
    else:
        st.caption("No uploaded files yet. Upload a document before asking questions.")

    if st.button("Clear chat", use_container_width=True):
        if st.session_state.conversation_id:
            direct_services()["agent"].memory.reset(st.session_state.conversation_id)
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.session_state.last_results = []
        st.rerun()

st.title("Local File Discovery Agent")
st.caption("Conversational search over the Assignment folder and uploaded files.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

pending = st.session_state.pop("pending_prompt", None)
user_prompt = pending or st.chat_input("Ask for files by topic, type, date, or contents...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    payload = {
        "message": user_prompt,
        "conversation_id": st.session_state.conversation_id,
        "search_mode": st.session_state.search_mode,
    }

    with st.chat_message("assistant"):
        with st.spinner("Searching files..."):
            try:
                response = backend_post("/chat", payload)
                response.raise_for_status()
                data = response.json()
                st.session_state.conversation_id = data.get("conversation_id")
                answer = data.get("answer", "No answer returned.")
                st.markdown(answer)
                results = data.get("results", [])
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.last_results = results
            except Exception as exc:
                error = f"Request failed: {exc}"
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})
