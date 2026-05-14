from __future__ import annotations

import logging
from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.agent.prompts import DRIVE_QUERY_SYSTEM_PROMPT
from app.config import Settings
from app.services.drive_service import DriveQueryBuilder

logger = logging.getLogger(__name__)


class DriveQueryPlan(BaseModel):
    q: str = Field(description="A valid Google Drive API v3 q parameter.")
    explanation: str = Field(description="Brief explanation of filters used.")
    confidence: float = Field(ge=0.0, le=1.0)
    search_strategy: Literal["name", "fullText", "metadata", "hybrid"]


class DriveQueryChain:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.fallback = DriveQueryBuilder(settings.google_drive_folder_id)
        self.llm = (
            ChatGroq(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                temperature=0.1,
                max_tokens=1024,
            )
            if settings.groq_api_key
            else None
        )

    def generate(self, query: str) -> DriveQueryPlan:
        if not self.llm:
            q, explanation = self.fallback.build(query)
            return DriveQueryPlan(q=q, explanation=explanation, confidence=0.55, search_strategy="hybrid")
        try:
            structured = self.llm.with_structured_output(DriveQueryPlan)
            plan = structured.invoke(
                [
                    ("system", DRIVE_QUERY_SYSTEM_PROMPT),
                    ("human", f"User search request: {query}"),
                ]
            )
            if "trashed=false" not in plan.q:
                plan.q = f"trashed=false and ({plan.q})"
            return plan
        except Exception:
            logger.exception("LLM Drive query generation failed; using fallback builder")
            q, explanation = self.fallback.build(query)
            return DriveQueryPlan(q=q, explanation=explanation, confidence=0.5, search_strategy="hybrid")
