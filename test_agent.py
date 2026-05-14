import asyncio
from frontend.streamlit_app import backend_post

payload = {
    "message": "perform ocr on image.png and return the extracted text",
    "search_mode": "hybrid",
}
response = backend_post("/chat", payload)
print(response.json())
