from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")


class Settings(BaseSettings):
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    google_application_credentials: str | None = Field(
        default=None, alias="GOOGLE_APPLICATION_CREDENTIALS"
    )
    google_service_account_json: str | None = Field(
        default=None, alias="GOOGLE_SERVICE_ACCOUNT_JSON"
    )
    google_service_account_json_b64: str | None = Field(
        default=None, alias="GOOGLE_SERVICE_ACCOUNT_JSON_B64"
    )
    google_drive_folder_id: str | None = Field(default=None, alias="GOOGLE_DRIVE_FOLDER_ID")
    fastapi_url: str = Field(default="http://localhost:8000", alias="FASTAPI_URL")
    local_assignment_path: str = Field(
        default=str(PROJECT_DIR / "Assignment"), alias="LOCAL_ASSIGNMENT_PATH"
    )
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    max_search_results: int = Field(default=10, alias="MAX_SEARCH_RESULTS")
    search_uploads_only: bool = Field(default=False, alias="SEARCH_UPLOADS_ONLY")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    pinecone_api_key: str | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index_name: str | None = Field(default=None, alias="PINECONE_INDEX_NAME")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    @property
    def local_assignment_dir(self) -> Path:
        path = Path(self.local_assignment_path).expanduser()
        if not path.is_absolute():
            path = BACKEND_DIR / path
        return path.resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
