# Conversational Local File Discovery Agent

Production-ready conversational file discovery over a local `Assignment/` folder, with uploads for new PDFs, Word documents, spreadsheets, CSVs, and text files.

The app uses FastAPI for the backend, LangChain tool calling with Groq for the agent, and Streamlit for the chat UI. It runs without Docker and is ready for Railway plus Streamlit Cloud deployment. Google Drive support still exists in the backend, but it is optional and not used by the local-only UI.

## Features

- Conversational multi-turn file search with memory
- Local uploaded-file search over PDF, DOCX, TXT, CSV, XLS, XLSX, images, and videos
- Upload new PDF, DOCX, TXT, CSV, XLS, XLSX, PNG, JPG, JPEG, WEBP, GIF, MP4, MOV, and M4V files into `Assignment/uploads/`
- Optional OCR for images and sampled video frames when Tesseract is available on the host
- Hybrid search mode across Drive and local files
- Dedicated LangChain tools: `DriveSearchTool`, `LocalSearchTool`, `FileSummaryTool`
- Query explanations, confidence scoring, suggested follow-ups, and search history
- Async FastAPI endpoints with Pydantic schemas, CORS, logging, and error handling
- Clean Streamlit chat UI with backend health, mode toggle, loading states, and reset

## Project Structure

```text
TAILOR_TALK_ASSIGNMENT/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── agent/
│   │   │   ├── graph.py
│   │   │   ├── prompts.py
│   │   │   ├── tools.py
│   │   │   ├── memory.py
│   │   │   └── chains.py
│   │   ├── services/
│   │   │   ├── drive_service.py
│   │   │   ├── local_search.py
│   │   │   └── embeddings.py
│   │   ├── models/
│   │   └── utils/
│   ├── requirements.txt
│   ├── railway.json
│   ├── nixpacks.toml
│   ├── Procfile
│   └── .env.example
├── frontend/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── .streamlit/config.toml
├── Assignment/
├── README.md
└── .gitignore
```

## Setup

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `backend/.env`:

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
FASTAPI_URL=http://localhost:8000
LOCAL_ASSIGNMENT_PATH=../Assignment
CORS_ORIGINS=*
SEARCH_UPLOADS_ONLY=false
```

Do not commit real API keys or service account JSON files.

### 2. Google Service Account Optional

You do not need Google Drive credentials for local search and uploads. Leave these blank unless you want to re-enable Drive mode:

```env
GOOGLE_APPLICATION_CREDENTIALS=
GOOGLE_DRIVE_FOLDER_ID=
```

If you do want Drive search later:

1. Open Google Cloud Console.
2. Create or select a project.
3. Enable **Google Drive API**.
4. Go to **IAM & Admin > Service Accounts**.
5. Create a service account.
6. Open the service account and create a JSON key.
7. Save the JSON file somewhere private.
8. Set `GOOGLE_APPLICATION_CREDENTIALS` to that JSON path.

### 3. Share Drive Folder Optional

1. Open the copied Google Drive assignment folder.
2. Click **Share**.
3. Add the service account email, usually ending with `iam.gserviceaccount.com`.
4. Give Viewer access.
5. Copy the folder ID from the URL and set `GOOGLE_DRIVE_FOLDER_ID`.

### 4. Groq API

Create a Groq API key in the Groq console and set `GROQ_API_KEY` in `.env`. The default model is:

```env
GROQ_MODEL=openai/gpt-oss-120b
```

## Run Locally

Start FastAPI:

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Start Streamlit:

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
FASTAPI_URL=http://localhost:8000 streamlit run streamlit_app.py
```

Open the Streamlit URL, upload files from the sidebar, and search conversationally.

By default, `SEARCH_UPLOADS_ONLY=false`, so the assistant searches both the original files in `Assignment/` and files uploaded through the app. Set it to `true` only if you want answers limited to uploaded files.

## API

### `GET /`

Basic service metadata.

### `GET /health`

Returns backend status and whether Groq, Google Drive, and the local folder are configured.

### `POST /chat`

Conversational agent endpoint.

```json
{
  "message": "Find PDFs about AI from last week",
  "conversation_id": null,
  "search_mode": "local"
}
```

### `POST /search`

Direct search endpoint.

```json
{
  "query": "Search documents containing revenue forecast",
  "search_mode": "local",
  "limit": 10
}
```

### `POST /upload`

Multipart upload endpoint used by the Streamlit sidebar.

```bash
curl -F "file=@Quarterly Forecast.pdf" http://localhost:8000/upload
```

### `GET /uploads`

Lists files currently in the uploaded searchable corpus.

## OCR And Video Processing

Images are processed with optional OCR:

- PNG, JPG, JPEG, WEBP, and GIF uploads are accepted.
- If `pytesseract`, `Pillow`, and the Tesseract system binary are available, extracted text becomes searchable.
- OCR uses upscaling, contrast enhancement, denoising, adaptive thresholding, and multiple Tesseract page segmentation modes.
- If OCR is unavailable, the file is still searchable by filename and metadata.
- Scanned PDFs are OCR'd with `pdf2image` when Poppler is available and normal embedded PDF text is empty.

Videos are processed conservatively:

- MP4, MOV, and M4V uploads are accepted.
- `opencv-python-headless` extracts duration, frame count, and FPS.
- A few frames are sampled for OCR when Tesseract is available.
- The app does not currently transcribe speech from video audio.

For local OCR on macOS:

```bash
brew install tesseract
brew install poppler
```

For Linux/Railway, add Tesseract and Poppler through the platform's package support if you need OCR in production.

## Google Drive Query Generation

The agent is prompted to generate Google Drive API v3 `q` clauses such as:

```text
trashed=false and mimeType='application/pdf' and name contains 'AI' and modifiedTime > '2026-05-07T00:00:00'
```

It understands:

- `name contains`
- `name='exact filename.ext'`
- `mimeType`
- `fullText contains`
- `modifiedTime`
- `createdTime`
- file type aliases such as PDFs, spreadsheets, documents, images, and slides

## Architecture

The backend creates shared services at startup:

- `GoogleDriveService`: optional service account auth, Drive query execution, pagination, metadata extraction
- `LocalSearchService`: upload handling, upload-only search scope, file traversal, text extraction, image OCR, video metadata/frame OCR, filename/content search, TF-IDF semantic fallback
- `DriveDiscoveryAgent`: LangChain Groq model with tool calling, memory, search refinement, final response generation
- `ConversationMemory`: in-memory multi-turn message and search history store

Search results are normalized into:

```json
{
  "name": "Daily Report.pdf",
  "type": "pdf",
  "modified_time": "2026-05-14T10:00:00",
  "link": "https://drive.google.com/...",
  "file_id": "abc123",
  "path": "reports/Daily Report.pdf",
  "confidence": 0.87,
  "summary": "Short extracted summary",
  "source": "google_drive"
}
```

## Railway Deployment

Deploy the FastAPI backend to Railway and the Streamlit UI separately. Railway should run only the backend service.

1. Push this repository to GitHub.
2. Create a new Railway project from the repo.
3. Keep the Railway service root directory as `/` so the top-level `Assignment/` corpus is included in the deployment.
4. Use the root `Dockerfile` builder.
5. Add environment variables:

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
SEARCH_UPLOADS_ONLY=false
CORS_ORIGINS=*
MAX_SEARCH_RESULTS=10
```

Leave `LOCAL_ASSIGNMENT_PATH` unset for the bundled top-level `Assignment/` folder. Set it only when using a Railway Volume or another custom absolute path.

6. If using Google Drive mode, also add:

```env
GOOGLE_DRIVE_FOLDER_ID=your_drive_folder_id
GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
```

For service account credentials, paste the full JSON key into `GOOGLE_SERVICE_ACCOUNT_JSON` as a secret variable. Do not commit the JSON key.

Railway will use the root `Dockerfile`, which starts:

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
```

After deployment, open:

```text
https://your-railway-backend.up.railway.app/health
```

The response should show `status: "ok"`, `groq_configured: true`, and `local_folder_exists: true`.

## Fly.io Deployment

Deploy the FastAPI backend from this repository using the root `Dockerfile`.

Recommended Fly settings:

- Builder: Dockerfile
- Dockerfile path: `/Dockerfile`
- Internal port: `8080`
- Healthcheck path: `/health`

Add these app secrets/environment variables:

```env
PORT=8080
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
FASTAPI_URL=http://localhost:8000
SEARCH_UPLOADS_ONLY=false
CORS_ORIGINS=*
MAX_SEARCH_RESULTS=10
LOG_LEVEL=INFO
```

For Google Drive search, also add:

```env
GOOGLE_DRIVE_FOLDER_ID=your_drive_folder_id
GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
```

`GOOGLE_APPLICATION_CREDENTIALS` is only for local file paths. Do not set it to a `/Users/...` path on Fly.io.
If your hosting provider's secret editor mangles multiline JSON, use `GOOGLE_SERVICE_ACCOUNT_JSON_B64` instead.

### Railway Inactivity

Railway Serverless/App Sleeping considers a service inactive after **10 minutes**, which is **600 seconds**, with no packets sent from the service. After the service sleeps, the next request wakes it up, but that first request may be slower because of cold start.

For demos, this is usually fine. For a smoother always-on experience, disable App Sleeping or use a plan/configuration that keeps the backend warm.

### Files And Persistence On Railway

Local search depends on files being present in `LOCAL_ASSIGNMENT_PATH` on the backend host.

- For an assignment demo, commit the sample `Assignment/` files to the repo so Railway deploys them with the backend.
- For user uploads, Railway's normal deployment filesystem may be temporary across redeploys. Attach a Railway Volume or move uploads to durable storage if uploaded files must persist.
- If you attach a volume, point `LOCAL_ASSIGNMENT_PATH` to the mounted directory and keep `SEARCH_UPLOADS_ONLY=false` or `true` depending on whether you want bundled demo files included.

## Streamlit Cloud Deployment

The app can run in a single Streamlit Cloud deployment. When `FASTAPI_URL` is blank, `direct`, or `http://localhost:8000`, the Streamlit app runs the backend agent in-process instead of calling a separate FastAPI server.

1. Deploy from GitHub.
2. Set the app path to `frontend/streamlit_app.py`.
3. Keep `FASTAPI_URL` unset, or set it to:

```env
FASTAPI_URL=http://localhost:8000
```

4. Add these secrets/environment variables:

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
SEARCH_UPLOADS_ONLY=false
CORS_ORIGINS=*
MAX_SEARCH_RESULTS=10
LOG_LEVEL=INFO
```

5. For Google Drive search, also add:

```env
GOOGLE_DRIVE_FOLDER_ID=your_drive_folder_id
GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
```

If TOML quoting causes credential errors, use a base64 value instead:

```toml
GOOGLE_SERVICE_ACCOUNT_JSON_B64 = "base64_encoded_service_account_json"
```

6. Deploy and confirm the sidebar health check is green.

If you later deploy FastAPI separately, set `FASTAPI_URL` to that backend URL and the Streamlit UI will call it over HTTP.

## Screenshot Placeholders

Add screenshots here after deployment:

- Chat UI with local search results
- Sidebar health and mode controls
- FastAPI docs at `/docs`

## Notes

- The current memory implementation is in-process. For multi-instance production, replace it with Redis or a database-backed chat history.
- Service account access only works for files/folders shared with the service account email.
- Local search depends on files being present in `LOCAL_ASSIGNMENT_PATH` on the backend host.
- Railway Serverless/App Sleeping inactivity is 600 seconds. Expect cold starts after sleep.
