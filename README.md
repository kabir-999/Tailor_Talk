# Tailor Talk - AI Drive File Discovery Agent 🚀

A powerful, context-aware AI agent that discovers, extracts, and summarizes content from both **Google Drive** and **local Assignment folders** using **Pinecone Vector RAG** and **Gemini 2.5 Flash**.

## 🌟 Key Features

- **Vector RAG Search:** Semantic document retrieval powered by Pinecone and Llama-text-embed-v2.
- **True Context Awareness:** The agent remembers extracted text and can answer follow-up questions about document content.
- **Hybrid Discovery:** Seamlessly searches across Google Drive and local filesystems.
- **Deep Content Extraction:** Supports PDF, DOCX, XLSX, and Image OCR.
- **Smart Cleanup:** Automatically cleans up "garbage" OCR artifacts for a professional reading experience.

## 🛠️ Technology Stack

- **Core:** Python, FastAPI, Streamlit
- **LLM:** Google Gemini 2.5 Flash
- **Vector DB:** Pinecone (Serverless)
- **Agent Framework:** LangGraph-inspired state orchestration
- **Search:** Hybrid semantic + keyword discovery

## 🚀 Setup & Installation

### 1. Environment Variables
Create a `.env` file in the `backend/` directory (see `.env.example` for details):

```env
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=llama-text-embed-v2-index
GOOGLE_DRIVE_FOLDER_ID=your_folder_id
GOOGLE_SERVICE_ACCOUNT_JSON={...}
```

### 2. Streamlit Cloud Secrets (TOML)
If deploying to Streamlit Cloud, you **must** use TOML format in the Secrets settings:

```toml
GEMINI_API_KEY = "..."
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "llama-text-embed-v2-index"
GOOGLE_SERVICE_ACCOUNT_JSON = '''
{
  "type": "service_account",
  ...
}
'''
```

### 3. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start the backend
uvicorn backend.app.main:app --reload

# Start the frontend
streamlit run frontend/streamlit_app.py
```

## 📝 Important Notes
- **Dependencies:** This project uses a zero-SDK Pinecone implementation via REST to ensure maximum compatibility with Streamlit Cloud.
- **Indexing:** New local files are automatically indexed into the vector database upon discovery or summary.
