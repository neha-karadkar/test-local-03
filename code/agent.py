import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

from config import Config

# ═══════════════════════════════════════════════════════════════════════════════
# Constants (from USER PROMPT TEMPLATE and AGENT DESIGN)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a professional knowledge retrieval agent specializing in general and healthcare domains. "
    "Your task is to answer user questions by searching the approved knowledge base documents using Azure AI Search. "
    "Retrieve the most relevant information from the filtered document(s) and generate a clear, concise, and accurate answer based strictly on the retrieved content. "
    "If the answer cannot be found in the provided context, politely inform the user that no relevant information is available. "
    "Do not speculate or provide advice beyond the retrieved information."
)
OUTPUT_FORMAT = "Provide a direct, well-structured answer in complete sentences. If the answer is not found, respond with the fallback message."
FALLBACK_RESPONSE = "I'm sorry, I could not find relevant information to answer your question based on the available knowledge base."

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

SELECTED_DOCUMENT_TITLES = ["Healthcare.pdf"]

ENRICHED_FIELDS = ["entities", "keyphrases", "relationships"]

TOP_K = 5

# ═══════════════════════════════════════════════════════════════════════════════
# Observability Lifespan Function
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI App Initialization
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(lifespan=_obs_lifespan,

    title="General Knowledge Healthcare Answer Agent",
    description="Answers user questions from Healthcare.pdf using Azure AI Search and GPT-4.1. Strict document filtering and professional fallback messaging.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

# ═══════════════════════════════════════════════════════════════════════════════
# Input Validation Models
# ═══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str = Field(..., description="User's question to be answered from the healthcare knowledge base.")

    @model_validator(mode="after")
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_query(self):
        if not self.query or not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("Query must be a non-empty string.")
        if len(self.query.strip()) > 50000:
            raise ValueError("Query exceeds maximum allowed length (50,000 characters).")
        return self

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the agent successfully answered the query.")
    answer: str = Field(..., description="Agent's answer or fallback message.")
    error: Optional[str] = Field(None, description="Error message if applicable.")
    fixing_tip: Optional[str] = Field(None, description="Helpful tip for fixing input errors.")
    trace_id: Optional[str] = Field(None, description="Observability trace ID for this request.")

# ═══════════════════════════════════════════════════════════════════════════════
# Error Handling: JSON Exception Handler
# ═══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle generic errors and malformed JSON input."""
    msg = str(exc)
    tip = None
    if "Expecting value" in msg or "JSONDecodeError" in msg:
        tip = "Malformed JSON. Please check your quotes, commas, and brackets."
    elif "field required" in msg:
        tip = "Missing required field. Please provide all necessary parameters."
    elif "Query must be a non-empty string" in msg:
        tip = "Your question cannot be empty. Please enter a valid question."
    elif "exceeds maximum allowed length" in msg:
        tip = "Your question is too long. Please shorten it to under 50,000 characters."
    else:
        tip = "Please check your input and try again."
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "answer": FALLBACK_RESPONSE,
            "error": msg,
            "fixing_tip": tip,
            "trace_id": None
        }
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Content Safety & Output Sanitization Utilities
# ═══════════════════════════════════════════════════════════════════════════════

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# ═══════════════════════════════════════════════════════════════════════════════
# AuditLogger
# ═══════════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """Logs agent actions, queries, responses, and errors for compliance and monitoring."""
    def __init__(self):
        self.logger = logging.getLogger("agent.audit")
        self.logger.setLevel(logging.INFO)

    def log_event(self, event: dict) -> None:
        try:
            self.logger.info(json.dumps(event, default=str))
        except Exception as e:
            self.logger.warning(f"Audit log failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# ErrorHandler
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorHandler:
    """Handles errors, timeouts, and fallback behaviors."""
    ERROR_MAP = {
        "KB_NOT_FOUND": FALLBACK_RESPONSE,
        "INVALID_QUERY": "Your question could not be processed. Please check your input.",
        "GENERIC_ERROR": "An unexpected error occurred. Please try again later."
    }

    def handle_error(self, error_code: str, context: dict = None) -> str:
        msg = self.ERROR_MAP.get(error_code, FALLBACK_RESPONSE)
        return msg

# ═══════════════════════════════════════════════════════════════════════════════
# ResponseFormatter
# ═══════════════════════════════════════════════════════════════════════════════

class ResponseFormatter:
    """Formats LLM output according to output_format instructions; applies fallback response if needed."""
    def format_response(self, answer: str, fallback: str) -> str:
        answer_clean = sanitize_llm_output(answer, content_type="text")
        if not answer_clean or answer_clean.strip() == "":
            return fallback
        return answer_clean

# ═══════════════════════════════════════════════════════════════════════════════
# ToolRegistry (empty for extensibility)
# ═══════════════════════════════════════════════════════════════════════════════

class BaseTool:
    """Abstract base class for tool integrations."""
    pass

class ToolRegistry:
    """Manages OpenAI function-calling tools (empty for this agent, extensible)."""
    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: BaseTool):
        pass  # No tools for this agent

    def execute_tool_call(self, tool_name: str, params: dict):
        pass  # No tools for this agent

# ═══════════════════════════════════════════════════════════════════════════════
# ChunkRetriever
# ═══════════════════════════════════════════════════════════════════════════════

class ChunkRetriever:
    """Queries Azure AI Search for relevant chunks using vector + keyword search, applies OData filter for selected documents."""

    _logger = logging.getLogger("agent.retrieval")
    _enriched_available = None  # None = not yet checked, True/False after first search

    def __init__(self):
        self.search_client = None
        self._init_client()

    def _init_client(self):
        if not self.search_client:
            self.search_client = SearchClient(
                endpoint=Config.AZURE_SEARCH_ENDPOINT,
                index_name=Config.AZURE_SEARCH_INDEX_NAME,
                credential=AzureKeyCredential(Config.AZURE_SEARCH_API_KEY),
            )

    async def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for the query using Azure OpenAI."""
        api_key = Config.AZURE_OPENAI_API_KEY
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not configured")
        client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        _t0 = _time.time()
        resp = await client.embeddings.create(
            input=query,
            model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
        )
        try:
            trace_tool_call(
                tool_name="openai_client.embeddings.create",
                latency_ms=int((_time.time() - _t0) * 1000),
                output=str(resp)[:200] if resp is not None else None,
                status="success",
            )
        except Exception:
            pass
        return resp.data[0].embedding

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, filter: Optional[str], top_k: int) -> List[dict]:
        """Retrieve relevant chunks from Azure AI Search with OData filter and enriched fields fallback."""
        embedding = await self._embed_query(query)
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="vector")
        base_fields = ["chunk", "title"]
        select_fields = base_fields + ENRICHED_FIELDS if self._enriched_available is not False else base_fields

        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": top_k,
            "select": select_fields,
        }
        if filter:
            search_kwargs["filter"] = filter

        from azure.core.exceptions import HttpResponseError
        _t0 = _time.time()
        try:
            results = list(self.search_client.search(**search_kwargs))
            if self._enriched_available is None:
                self._enriched_available = True
                self._logger.info("Enriched index fields are AVAILABLE — using: %s", ENRICHED_FIELDS)
            try:
                trace_tool_call(
                    tool_name="search_client.search",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    output=str(results)[:200] if results is not None else None,
                    status="success",
                )
            except Exception:
                pass
            return results
        except HttpResponseError as e:
            if "Could not find a property named" in str(e) and self._enriched_available is not False:
                self._enriched_available = False
                self._logger.warning("Enriched index fields NOT available in this index — falling back to base fields: %s", base_fields)
                search_kwargs["select"] = base_fields
                results = list(self.search_client.search(**search_kwargs))
                try:
                    trace_tool_call(
                        tool_name="search_client.search",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        output=str(results)[:200] if results is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                return results
            raise

    def build_filter(self, selected_titles: List[str]) -> Optional[str]:
        """Build OData filter for selected document titles."""
        if not selected_titles:
            return None
        odata_parts = [f"title eq '{t}'" for t in selected_titles]
        return " or ".join(odata_parts)

# ═══════════════════════════════════════════════════════════════════════════════
# LLMService
# ═══════════════════════════════════════════════════════════════════════════════

class LLMService:
    """Calls Azure OpenAI GPT-4.1 with enhanced system prompt, user query, and retrieved chunks as context."""

    def __init__(self):
        self.client = None

    def _init_client(self):
        if not self.client:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            self.client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_answer(self, prompt: str, context: List[str], user_query: str) -> str:
        """Generate answer from LLM using system prompt, context, and user query."""
        self._init_client()
        system_message = prompt + "\n\nOutput Format: " + OUTPUT_FORMAT
        context_text = "\n\n".join(context)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query},
            {"role": "user", "content": context_text}
        ]
        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        response = await self.client.chat.completions.create(
            model=Config.LLM_MODEL or "gpt-4.1",
            messages=messages,
            **_llm_kwargs
        )
        content = response.choices[0].message.content
        try:
            trace_model_call(
                provider="azure",
                model_name=Config.LLM_MODEL or "gpt-4.1",
                prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary=content[:200] if content else "",
            )
        except Exception:
            pass
        return content

# ═══════════════════════════════════════════════════════════════════════════════
# HealthcareKnowledgeAgent
# ═══════════════════════════════════════════════════════════════════════════════

class HealthcareKnowledgeAgent:
    """Coordinates input processing, retrieval, LLM interaction, response generation, error handling, and logging."""

    def __init__(self):
        self.chunk_retriever = ChunkRetriever()
        self.llm_service = LLMService()
        self.response_formatter = ResponseFormatter()
        self.error_handler = ErrorHandler()
        self.audit_logger = AuditLogger()
        self.tool_registry = ToolRegistry()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_user_query(self, query: str) -> Dict[str, Any]:
        """Entry point for handling user queries; orchestrates retrieval, LLM interaction, response formatting, and error handling."""
        async with trace_step(
            "parse_input", step_type="parse",
            decision_summary="Validate and parse user query",
            output_fn=lambda r: f"query={r}",
        ) as step:
            parsed_query = query.strip()
            step.capture(parsed_query)

        filter_str = self.chunk_retriever.build_filter(SELECTED_DOCUMENT_TITLES)
        async with trace_step(
            "retrieve_chunks", step_type="process",
            decision_summary="Retrieve relevant chunks from Azure AI Search",
            output_fn=lambda r: f"chunks_found={len(r)}",
        ) as step:
            try:
                results = await self.chunk_retriever.retrieve_chunks(parsed_query, filter_str, TOP_K)
                step.capture(results)
            except Exception as e:
                self.audit_logger.log_event({
                    "event": "retrieval_error",
                    "query": parsed_query,
                    "error": str(e)
                })
                answer = self.error_handler.handle_error("KB_NOT_FOUND")
                return {
                    "success": False,
                    "answer": answer,
                    "error": str(e),
                    "fixing_tip": "Please rephrase your question or try again later.",
                    "trace_id": None
                }

        # Format context for LLM: include enriched fields as metadata
        context_parts = []
        enriched_available = self.chunk_retriever._enriched_available
        for r in results:
            part = r.get("chunk", "")
            if enriched_available:
                for field in ENRICHED_FIELDS:
                    value = r.get(field)
                    if value:
                        part += f"\n{field}: {json.dumps(value) if isinstance(value, (list, dict)) else value}"
            context_parts.append(part)

        async with trace_step(
            "llm_interaction", step_type="llm_call",
            decision_summary="Generate answer from LLM using context",
            output_fn=lambda r: f"answer={r[:80]}",
        ) as step:
            try:
                answer_raw = await self.llm_service.generate_answer(
                    SYSTEM_PROMPT, context_parts, parsed_query
                )
                step.capture(answer_raw)
            except Exception as e:
                self.audit_logger.log_event({
                    "event": "llm_error",
                    "query": parsed_query,
                    "error": str(e)
                })
                answer = self.error_handler.handle_error("GENERIC_ERROR")
                return {
                    "success": False,
                    "answer": answer,
                    "error": str(e),
                    "fixing_tip": "Please try again later.",
                    "trace_id": None
                }

        async with trace_step(
            "response_generation", step_type="format",
            decision_summary="Format LLM output and apply fallback if needed",
            output_fn=lambda r: f"final_answer={r[:80]}",
        ) as step:
            answer_final = self.response_formatter.format_response(answer_raw, FALLBACK_RESPONSE)
            step.capture(answer_final)

        self.audit_logger.log_event({
            "event": "query_answered",
            "query": parsed_query,
            "answer": answer_final,
            "chunks_count": len(context_parts),
            "trace_id": None  # Trace ID can be injected from observability if needed
        })

        return {
            "success": True,
            "answer": answer_final,
            "error": None,
            "fixing_tip": None,
            "trace_id": None
        }

# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint(req: QueryRequest):
    agent = HealthcareKnowledgeAgent()
    result = await agent.process_user_query(req.query)
    # Sanitize output before returning
    result["answer"] = sanitize_llm_output(result.get("answer", ""), content_type="text")
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# Entrypoint (_run_agent) and __main__ block
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())