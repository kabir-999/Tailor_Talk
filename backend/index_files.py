import os
from pathlib import Path
from dotenv import load_dotenv
from app.config import get_settings
from app.services.local_search import LocalSearchService

load_dotenv()
settings = get_settings()
local_service = LocalSearchService(settings.local_assignment_dir, settings=settings, uploads_only=False)

print(f"Indexing files in {settings.local_assignment_dir}...")
files = local_service._iter_files()
for path in files:
    print(f"Indexing {path.name}...")
    # summarize_file triggers upsert_document
    local_service.summarize_file(str(path), max_chars=1)

print("Done!")
