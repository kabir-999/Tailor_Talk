from __future__ import annotations

import html
import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BACKEND = os.getenv("FASTAPI_URL", "http://localhost:8000").rstrip("/")

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
    st.session_state.setdefault("backend_url", DEFAULT_BACKEND)
    st.session_state.setdefault("search_mode", "local")
    st.session_state.setdefault("last_results", [])


def backend_get(path: str) -> requests.Response:
    return requests.get(f"{st.session_state.backend_url}{path}", timeout=8)


def backend_post(path: str, payload: dict[str, Any]) -> requests.Response:
    return requests.post(f"{st.session_state.backend_url}{path}", json=payload, timeout=180)


def backend_upload(file: Any) -> requests.Response:
    files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
    return requests.post(f"{st.session_state.backend_url}/upload", files=files, timeout=90)


def backend_delete_upload(path: str) -> requests.Response:
    return requests.delete(f"{st.session_state.backend_url}/uploads/{path}", timeout=30)


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
    st.session_state.backend_url = st.text_input("Backend URL", st.session_state.backend_url).rstrip("/")

    health_placeholder = st.empty()
    try:
        health = backend_get("/health").json()
        health_placeholder.success("Backend healthy")
        st.caption(f"Groq: {'configured' if health.get('groq_configured') else 'missing'}")
        st.caption(f"Local folder: {'found' if health.get('local_folder_exists') else 'missing'}")
        st.caption(f"Uploaded files: {health.get('uploaded_file_count', 0)}")
        st.caption(f"Folder: {health.get('local_folder', '')}")
    except Exception:
        health_placeholder.error("Backend unavailable")

    mode_labels = {
        "local": "Local files",
        "drive": "Google Drive",
        "hybrid": "Local + Drive",
    }
    selected_mode = st.radio(
        "Search scope",
        options=list(mode_labels.keys()),
        format_func=mode_labels.get,
    )
    st.session_state.search_mode = selected_mode

    st.divider()
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
