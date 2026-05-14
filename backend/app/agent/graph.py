from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq

from app.agent.chains import DriveQueryChain
from app.agent.memory import ConversationMemory
from app.agent.prompts import AGENT_SYSTEM_PROMPT, FINAL_RESPONSE_PROMPT
from app.agent.tools import build_tools
from app.config import Settings
from app.models.schemas import ChatResponse, FileResult, SearchMode, SearchResponse
from app.services.drive_service import GoogleDriveService
from app.services.local_search import LocalSearchService

logger = logging.getLogger(__name__)


class DriveDiscoveryAgent:
    def __init__(
        self,
        settings: Settings,
        drive_service: GoogleDriveService,
        local_service: LocalSearchService,
        memory: ConversationMemory,
    ) -> None:
        self.settings = settings
        self.drive_service = drive_service
        self.local_service = local_service
        self.memory = memory
        self.drive_query_chain = DriveQueryChain(settings)
        self.tools = build_tools(drive_service, local_service, self.drive_query_chain)
        self.tools_by_name: dict[str, StructuredTool] = {tool.name: tool for tool in self.tools}
        self.llm = (
            ChatGroq(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                temperature=0.2,
                max_tokens=2048,
            )
            if settings.groq_api_key
            else None
        )

    async def chat(self, message: str, conversation_id: str | None, search_mode: SearchMode) -> ChatResponse:
        conversation_id = conversation_id or self.memory.new_id()
        return await asyncio.to_thread(self._chat_sync, message, conversation_id, search_mode)

    async def search(self, query: str, search_mode: SearchMode, limit: int) -> SearchResponse:
        return await asyncio.to_thread(self._search_sync, query, search_mode, limit)

    def _chat_sync(self, message: str, conversation_id: str, search_mode: SearchMode) -> ChatResponse:
        self.memory.append_user(conversation_id, message)
        tool_payloads: list[dict[str, Any]] = []
        previous_results = self.memory.last_results(conversation_id)

        if self._is_text_cleanup_request(message):
            answer = self._cleanup_text_answer(message, conversation_id)
            self.memory.append_ai(conversation_id, answer)
            return ChatResponse(
                conversation_id=conversation_id,
                answer=answer,
                results=[],
                query_explanation="Cleaned text from the current conversation without running file search.",
                suggested_followups=[],
                search_history=self.memory.search_history(conversation_id),
            )

        if self._is_context_followup(message) and previous_results:
            tool_payloads = [
                {
                    "tool": "ConversationContext",
                    "query_explanation": "Used previous search results from this conversation.",
                    "results": previous_results,
                }
            ]

        if not tool_payloads and self.llm:
            try:
                tool_payloads = self._run_llm_tool_round(message, conversation_id, search_mode)
            except Exception:
                logger.exception("LLM tool round failed; using deterministic search fallback")

        if not tool_payloads:
            tool_payloads = [self._search_payload(message, search_mode, self.settings.max_search_results)]

        results = self._collect_results(tool_payloads)
        query_explanation = self._combine_explanations(tool_payloads)
        if results:
            self.memory.set_last_results(conversation_id, [result.model_dump() for result in results])
        if self._is_content_request(message) and results:
            answer = self._content_answer(message, conversation_id, results, previous_results)
            self.memory.append_ai(conversation_id, answer)
            self.memory.add_search(
                conversation_id,
                {
                    "message": message,
                    "mode": search_mode.value,
                    "result_count": len(results),
                    "query_explanation": query_explanation,
                },
            )
            return ChatResponse(
                conversation_id=conversation_id,
                answer=answer,
                results=results,
                query_explanation=query_explanation,
                suggested_followups=[],
                search_history=self.memory.search_history(conversation_id),
            )
        answer = self._final_answer(message, search_mode, tool_payloads, results, previous_results)
        self.memory.append_ai(conversation_id, answer)
        self.memory.add_search(
            conversation_id,
            {
                "message": message,
                "mode": search_mode.value,
                "result_count": len(results),
                "query_explanation": query_explanation,
            },
        )
        return ChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            results=results,
            query_explanation=query_explanation,
            suggested_followups=[],
            search_history=self.memory.search_history(conversation_id),
        )

    def _search_sync(self, query: str, search_mode: SearchMode, limit: int) -> SearchResponse:
        payload = self._search_payload(query, search_mode, limit)
        results = self._collect_results([payload])
        return SearchResponse(
            query=query,
            mode=search_mode,
            results=results,
            query_explanation=payload.get("query_explanation"),
            generated_drive_q=payload.get("generated_drive_q"),
        )

    def _run_llm_tool_round(self, message: str, conversation_id: str, search_mode: SearchMode) -> list[dict[str, Any]]:
        assert self.llm is not None
        mode_instruction = (
            f"Current UI search mode: {search_mode.value}. Respect it unless the user explicitly asks otherwise."
        )
        llm_with_tools = self.llm.bind_tools(self.tools)
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            SystemMessage(content=mode_instruction),
            SystemMessage(content=self._context_message(conversation_id)),
            *self.memory.get_messages(conversation_id),
        ]
        ai_message = llm_with_tools.invoke(messages)
        if not isinstance(ai_message, AIMessage) or not ai_message.tool_calls:
            return []

        payloads: list[dict[str, Any]] = []
        tool_messages: list[ToolMessage] = []
        for call in ai_message.tool_calls:
            tool = self.tools_by_name.get(call["name"])
            if not tool:
                continue
            try:
                output = tool.invoke(call.get("args", {}))
                parsed = json.loads(output)
                payloads.append(parsed)
                tool_messages.append(ToolMessage(content=output, tool_call_id=call["id"]))
            except Exception as exc:
                logger.exception("Tool call failed: %s", call)
                error_payload = {"tool": call["name"], "error": str(exc), "results": []}
                payloads.append(error_payload)
                tool_messages.append(ToolMessage(content=json.dumps(error_payload), tool_call_id=call["id"]))

        return payloads

    def _search_payload(self, query: str, search_mode: SearchMode, limit: int) -> dict[str, Any]:
        payloads: list[dict[str, Any]] = []
        if search_mode in {SearchMode.drive, SearchMode.hybrid}:
            try:
                plan = self.drive_query_chain.generate(query)
                drive_results, explanation, generated_q = self.drive_service.search(query, plan.q, limit)
                payloads.append(
                    {
                        "tool": "DriveSearchTool",
                        "query_explanation": f"{plan.explanation} {explanation}".strip(),
                        "generated_drive_q": generated_q,
                        "results": [result.model_dump() for result in drive_results],
                    }
                )
            except Exception as exc:
                logger.warning("Drive search unavailable: %s", exc)
                payloads.append({"tool": "DriveSearchTool", "error": str(exc), "results": []})

        if search_mode in {SearchMode.local, SearchMode.hybrid}:
            try:
                local_results, explanation = self.local_service.search(query, limit)
                payloads.append(
                    {
                        "tool": "LocalSearchTool",
                        "query_explanation": explanation,
                        "results": [result.model_dump() for result in local_results],
                    }
                )
            except Exception as exc:
                logger.warning("Local search unavailable: %s", exc)
                payloads.append({"tool": "LocalSearchTool", "error": str(exc), "results": []})

        merged = {
            "tool": "HybridSearch" if len(payloads) > 1 else payloads[0].get("tool", "Search"),
            "query_explanation": self._combine_explanations(payloads),
            "generated_drive_q": next((p.get("generated_drive_q") for p in payloads if p.get("generated_drive_q")), None),
            "results": [result for payload in payloads for result in payload.get("results", [])],
            "errors": [payload["error"] for payload in payloads if payload.get("error")],
        }
        merged["results"] = sorted(merged["results"], key=lambda item: item.get("confidence", 0), reverse=True)[:limit]
        return merged

    def _final_answer(
        self,
        message: str,
        search_mode: SearchMode,
        tool_payloads: list[dict[str, Any]],
        results: list[FileResult],
        previous_results: list[dict[str, Any]] | None = None,
    ) -> str:
        if self.llm:
            try:
                response = self.llm.invoke(
                    [
                        SystemMessage(content=FINAL_RESPONSE_PROMPT),
                        HumanMessage(
                            content=json.dumps(
                                {
                                    "user_message": message,
                                    "search_mode": search_mode.value,
                                    "tool_outputs": tool_payloads,
                                    "previous_results": previous_results or [],
                                },
                                default=str,
                            )
                        ),
                    ]
                )
                return str(response.content)
            except Exception:
                logger.exception("Final answer LLM call failed")

        if not results:
            errors = [error for payload in tool_payloads for error in payload.get("errors", [])]
            suffix = f" I also hit: {'; '.join(errors)}" if errors else ""
            return f"I could not find matching files for '{message}'.{suffix}\n\nNext prompt: \"Search with a broader keyword\""
        lines = [f"I found {len(results)} matching file(s):"]
        for result in results[:3]:
            locator = result.link or result.path or result.file_id or ""
            lines.append(f"- {result.name} ({result.source}, {result.type}) confidence {result.confidence:.2f} {locator}")
        lines.append("")
        lines.append(f"Next prompt: \"{DriveDiscoveryAgent._next_prompt(message, results)}\"")
        return "\n".join(lines)

    def _context_message(self, conversation_id: str) -> str:
        previous = self.memory.last_results(conversation_id)
        if not previous:
            return "No previous file results are available for this conversation."
        compact = [
            {
                "name": item.get("name"),
                "path": item.get("path"),
                "link": item.get("link"),
                "type": item.get("type"),
                "source": item.get("source"),
            }
            for item in previous[:5]
        ]
        return "Previous file results for follow-up references: " + json.dumps(compact)

    @staticmethod
    def _is_context_followup(message: str) -> bool:
        lowered = message.lower().strip()
        context_words = (
            "it",
            "that",
            "this",
            "those",
            "them",
            "previous",
            "above",
            "same",
            "clean",
            "summarize",
            "summary",
            "open",
            "show source",
            "ocr output",
        )
        return any(word in lowered for word in context_words) and not any(
            word in lowered for word in ("find", "search", "list all", "show all")
        )

    @staticmethod
    def _is_text_cleanup_request(message: str) -> bool:
        lowered = message.lower()
        cleanup_words = ("clean", "format", "properly format", "fix this", "rewrite this", "clean the ocr")
        return any(word in lowered for word in cleanup_words) and not any(
            word in lowered for word in ("find", "search", "list all", "show files")
        )

    @staticmethod
    def _is_content_request(message: str) -> bool:
        lowered = message.lower()
        content_words = (
            "ocr",
            "read",
            "extract",
            "summarize",
            "summary",
            "what is written",
            "what's written",
            "whats written",
            "what is inside",
            "what's inside",
            "whats inside",
            "what does",
            "what says",
            "content",
            "inside it",
        )
        return any(word in lowered for word in content_words)

    def _content_answer(
        self,
        message: str,
        conversation_id: str,
        results: list[FileResult],
        previous_results: list[dict[str, Any]] | None = None,
    ) -> str:
        result = self._best_content_result(message, results, previous_results)
        if result.source != "local" or not result.path:
            locator = result.link or result.file_id or result.name
            return (
                f"I found {result.name}, but I can only extract readable contents from local files in this deployment. "
                f"Open it here: {locator}\n\nNext prompt: \"Search local files for {result.name}\""
            )

        extracted = self.local_service.summarize_file(result.path, max_chars=3000)
        self.memory.set_last_extracted_text(conversation_id, extracted)
        if self.llm and any(word in message.lower() for word in ("summarize", "summary")):
            try:
                response = self.llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                "Summarize the extracted file text directly. Do not list matching files. "
                                "Do not add follow-up suggestions except the final required next prompt."
                            )
                        ),
                        HumanMessage(content=f"User request: {message}\n\nFile: {result.name}\n\nExtracted text:\n{extracted}"),
                    ]
                )
                summary = str(response.content).strip()
                return f"{summary}\n\nNext prompt: \"Show the full extracted text\""
            except Exception:
                logger.exception("Content summary LLM call failed")

        return (
            f"Extracted text from {result.name}:\n\n"
            f"{extracted}\n\n"
            f"Next prompt: \"Clean and format this extracted text\""
        )

    @staticmethod
    def _best_content_result(
        message: str,
        results: list[FileResult],
        previous_results: list[dict[str, Any]] | None = None,
    ) -> FileResult:
        lowered = message.lower()
        candidates = results
        filename_match = re.search(
            r"([\w][\w .-]*?\.(?:pdf|docx|txt|csv|xlsx|xls|png|jpg|jpeg|webp|gif|mp4|mov|m4v))",
            lowered,
            flags=re.IGNORECASE,
        )
        if filename_match:
            requested = re.sub(r"\s+", " ", filename_match.group(1).casefold()).strip()
            for result in sorted(candidates, key=lambda item: item.source != "local"):
                name = re.sub(r"\s+", " ", result.name.casefold()).strip()
                path = re.sub(r"\s+", " ", (result.path or "").casefold()).strip()
                if requested in name or requested in path:
                    return result

        if previous_results:
            for item in previous_results:
                previous = item if isinstance(item, FileResult) else FileResult(**item)
                if previous.name.lower() in lowered or (previous.path and previous.path.lower() in lowered):
                    return previous
        return candidates[0]

    def _cleanup_text_answer(self, message: str, conversation_id: str) -> str:
        text = self._cleanup_source_text(message, conversation_id)
        if not text.strip():
            return 'I do not have text to clean yet.\n\nNext prompt: "Paste the OCR text to clean"'

        if self.llm:
            try:
                response = self.llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                "Clean and properly format noisy OCR/chat text. "
                                "Do not search for files. Do not mention missing files. "
                                "Preserve useful factual content, remove obvious OCR garbage, "
                                "fix spacing and line breaks, and return only the cleaned text."
                            )
                        ),
                        HumanMessage(content=text),
                    ]
                )
                cleaned = str(response.content).strip()
                return f"{cleaned}\n\nNext prompt: \"Summarize this cleaned text\""
            except Exception:
                logger.exception("Text cleanup LLM call failed")

        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"([.!?])\s+", r"\1\n", cleaned)
        return f"{cleaned}\n\nNext prompt: \"Summarize this cleaned text\""

    def _cleanup_source_text(self, message: str, conversation_id: str) -> str:
        extracted_text = self.memory.last_extracted_text(conversation_id)
        if extracted_text.strip():
            return extracted_text

        lowered = message.lower()
        markers = (
            "i mean clean and properly format this",
            "clean and properly format this",
            "clean this",
            "format this",
            "clean the ocr output",
        )
        for marker in markers:
            index = lowered.find(marker)
            if index >= 0:
                before = message[:index].strip()
                after = message[index + len(marker) :].strip()
                if len(before) > 20:
                    return before
                if len(after) > 20:
                    return after

        messages = self.memory.get_messages(conversation_id)
        for previous in reversed(messages[:-1]):
            content = str(previous.content).strip()
            if len(content) > 20 and "Next prompt:" not in content:
                return content
        for previous in reversed(messages[:-1]):
            content = str(previous.content).strip()
            if len(content) > 20:
                return content
        return ""

    @staticmethod
    def _collect_results(payloads: list[dict[str, Any]]) -> list[FileResult]:
        seen: set[tuple[str | None, str | None, str]] = set()
        results: list[FileResult] = []
        for payload in payloads:
            for item in payload.get("results", []):
                result = item if isinstance(item, FileResult) else FileResult(**item)
                key = (result.file_id, result.path, result.source)
                if key not in seen:
                    seen.add(key)
                    results.append(result)
        return sorted(results, key=lambda result: result.confidence, reverse=True)

    @staticmethod
    def _combine_explanations(payloads: list[dict[str, Any]]) -> str | None:
        explanations = [payload.get("query_explanation") for payload in payloads if payload.get("query_explanation")]
        return " | ".join(explanations) if explanations else None

    @staticmethod
    def _suggest_followups(message: str, results: list[FileResult], search_mode: SearchMode) -> list[str]:
        if not results:
            return [
                f"Search {search_mode.value} for a broader keyword",
                "Try filtering by PDF, spreadsheet, document, or image",
                "Try a date range like modified this week or uploaded in April",
            ]
        top_type = results[0].type
        return [
            f"Show only {top_type} results",
            f"Find files like {results[0].name}",
            f"Search within these results for another keyword from: {message}",
        ]

    @staticmethod
    def _next_prompt(message: str, results: list[FileResult]) -> str:
        lowered = message.lower()
        if any(word in lowered for word in ("ocr", "scanned", "image", "screenshot")):
            return "Clean the OCR output"
        if any(word in lowered for word in ("summarize", "summary")):
            return "Show the source files"
        if results:
            return f"Summarize {results[0].name}"
        return "Search with a broader keyword"
