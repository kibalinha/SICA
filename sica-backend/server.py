import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ai_engine import get_ai_engine
from audio_processor import process_audio_file

load_dotenv()


class JSONFormatter(logging.Formatter):
    """Formatador JSON estruturado para logs."""

    _RESERVED_ATTRS: ClassVar[frozenset[str]] = frozenset(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    )

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Campos passados via extra={} viram atributos diretos do LogRecord
        for key, value in record.__dict__.items():
            if key not in self._RESERVED_ATTRS:
                try:
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = repr(value)

        return json.dumps(log_data, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    """Configura logging estruturado em JSON."""
    logger = logging.getLogger("sica")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("sica_backend.log")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()

MAX_UPLOAD_BYTES = int(os.getenv("SICA_MAX_UPLOAD_MB", "10")) * 1024 * 1024
DEFAULT_CORS_ORIGINS = (
    "http://localhost:4173,http://127.0.0.1:4173,http://localhost:5173,http://127.0.0.1:5173"
)
RATE_LIMIT_REQUESTS = int(os.getenv("SICA_RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_PERIOD = int(os.getenv("SICA_RATE_LIMIT_PERIOD", "60"))
ENABLE_CACHE = os.getenv("SICA_ENABLE_CACHE", "true").lower() == "true"
CACHE_SIZE = int(os.getenv("SICA_CACHE_SIZE", "100"))
CACHE_TTL_SECONDS = int(os.getenv("SICA_CACHE_TTL", "3600"))
API_VERSION = "v1"
API_KEY = os.getenv("SICA_API_KEY")

limiter = Limiter(key_func=get_remote_address)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_audio_cache: dict[str, dict[str, Any]] = {}
_cache_timestamps: dict[str, float] = {}


def _get_cache_key(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()


def _clean_expired_cache() -> None:
    if not ENABLE_CACHE:
        return
    now = time.time()
    expired_keys = [
        key for key, timestamp in _cache_timestamps.items() if now - timestamp > CACHE_TTL_SECONDS
    ]
    for key in expired_keys:
        _audio_cache.pop(key, None)
        _cache_timestamps.pop(key, None)
    if expired_keys:
        logger.debug(
            "cache_cleanup",
            extra={"removed_count": len(expired_keys), "remaining": len(_audio_cache)},
        )


def _get_from_cache(cache_key: str) -> dict[str, Any] | None:
    _clean_expired_cache()
    if ENABLE_CACHE and cache_key in _audio_cache:
        logger.debug("cache_hit", extra={"key_prefix": cache_key[:8]})
        return _audio_cache[cache_key]
    return None


def _set_cache(cache_key: str, result: dict[str, Any]) -> None:
    if not ENABLE_CACHE:
        return
    _clean_expired_cache()
    if len(_audio_cache) >= CACHE_SIZE:
        oldest_key = min(_cache_timestamps, key=lambda k: _cache_timestamps[k])
        _audio_cache.pop(oldest_key, None)
        _cache_timestamps.pop(oldest_key, None)
        logger.debug("cache_evict", extra={"key_prefix": oldest_key[:8]})

    _audio_cache[cache_key] = result
    _cache_timestamps[cache_key] = time.time()
    logger.debug("cache_set", extra={"key_prefix": cache_key[:8], "size": len(_audio_cache)})


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("SICA_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    if not raw or raw.strip() == "":
        return []
    if raw.strip() == "*":
        return ["*"]
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    # Remove duplicates while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for x in origins:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _validate_environment() -> None:
    try:
        if MAX_UPLOAD_BYTES <= 0:
            raise ValueError("SICA_MAX_UPLOAD_MB deve ser positivo")
        if MAX_UPLOAD_BYTES > 100 * 1024 * 1024:
            logger.warning(
                "config_warning",
                extra={"event": "SICA_MAX_UPLOAD_MB muito alto, considere reduzir"},
            )
        # Validate CORS origins
        cors_origins = _parse_cors_origins()
        for origin in cors_origins:
            if origin != "*" and not origin.startswith(("http://", "https://")):
                logger.warning(
                    "config_warning",
                    extra={"event": "SICA_CORS_ORIGINS contains invalid origin", "origin": origin},
                )
    except Exception as e:
        logger.exception("env_validation_error", extra={"error": str(e)})
        raise


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).stem
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    safe_name = "".join(c if c in safe_chars else "_" for c in name)
    safe_name = safe_name.replace(".", "")
    return safe_name[:100]


async def verify_api_key(api_key: str | None = Depends(_api_key_header)) -> None:
    """Verifica API Key se configurada."""
    if API_KEY:
        if not api_key:
            logger.warning("auth_failed", extra={"provided_key_prefix": "none"})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API Key ausente",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        if api_key != API_KEY:
            logger.warning(
                "auth_failed", extra={"provided_key_prefix": api_key[:8] if api_key else "none"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API Key inválida",
                headers={"WWW-Authenticate": "ApiKey"},
            )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("startup", extra={"event": "Iniciando SICA Backend...", "version": "1.2.0"})
    _validate_environment()
    try:
        get_ai_engine()
        logger.info("startup", extra={"event": "Motor de IA carregado com sucesso"})
    except Exception as e:
        logger.error("startup_error", extra={"error": str(e)}, exc_info=True)
        raise
    logger.info("startup", extra={"event": "SICA Backend pronto para aceitar requisições"})
    yield
    logger.info("shutdown", extra={"event": "Desligando SICA Backend..."})


app = FastAPI(
    title="SICA Backend - API de Análise Espectral e IA de Áudio",
    version="1.2.0",
    lifespan=lifespan,
    docs_url=f"/api/{API_VERSION}/docs",
    redoc_url=f"/api/{API_VERSION}/redoc",
    openapi_url=f"/api/{API_VERSION}/openapi.json",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

cors_origins = _parse_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_error", extra={"path": request.url.path, "error": str(exc)}, exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erro interno do servidor. Contate o administrador."},
    )


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "status": "SICA Audio API is running",
        "version": "1.2.0",
        "api_version": API_VERSION,
        "docs": f"/api/{API_VERSION}/docs",
    }


@app.get(f"/api/{API_VERSION}/health")
def health_check() -> dict[str, Any]:
    import psutil  # noqa: PLC0415  (lazy: evita carregar em todo import do módulo)
    import torch  # noqa: PLC0415

    engine = get_ai_engine()

    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.cwd().anchor or "."))

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.2.0",
        "api_version": API_VERSION,
        "ai_engine": "ready",
        "device": str(engine.device),
        "system": {
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
        },
        "model": {
            "loaded": True,
            "device": str(engine.device),
            "cuda_available": torch.cuda.is_available(),
        },
        "cache": {
            "enabled": ENABLE_CACHE,
            "size": len(_audio_cache),
            "max_size": CACHE_SIZE,
            "ttl_seconds": CACHE_TTL_SECONDS,
        },
    }


@app.post(f"/api/{API_VERSION}/analyze")
@limiter.limit(f"{RATE_LIMIT_REQUESTS}/{RATE_LIMIT_PERIOD} seconds")
async def analyze_audio(
    request: Request,
    file: UploadFile = File(...),
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
    client_ip = get_remote_address(request)
    logger.info("analyze_request", extra={"client_ip": client_ip})

    if not file.filename:
        logger.warning("analyze_error", extra={"error": "no_filename"})
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    safe_filename = _sanitize_filename(file.filename)
    logger.info("analyze_file_received", extra={"file_name": safe_filename})

    content = await file.read()
    if not content:
        logger.warning("analyze_error", extra={"error": "empty_file"})
        raise HTTPException(status_code=400, detail="Arquivo de áudio vazio.")
    if len(content) > MAX_UPLOAD_BYTES:
        logger.warning(
            "analyze_error", extra={"error": "file_too_large", "size_bytes": len(content)}
        )
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo excede o limite de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    suffix = Path(file.filename).suffix if file.filename else ""
    valid_exts = [".wav", ".mp3", ".ogg", ".flac", ".webm", ".m4a", ".mp4", ".aac"]
    if not suffix or suffix.lower() not in valid_exts:
        suffix = ".webm" if (file.content_type and "webm" in file.content_type) else ".wav"
        logger.info("analyze_extension_fallback", extra={"suffix": suffix})

    tmp_path = None
    start_time = time.time()
    try:
        cache_key = _get_cache_key(content)
        cached_result = _get_from_cache(cache_key)
        if cached_result:
            logger.info("analyze_cache_hit", extra={"file_name": safe_filename})
            return cached_result

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            logger.debug("analyze_temp_file", extra={"path": tmp_path})

        # process_audio_file é CPU-bound (STFT/HPSS/PyTorch) e bloquearia o
        # event loop do asyncio, travando todas as requisições concorrentes;
        # executa em threadpool para manter a API responsiva.
        result = await run_in_threadpool(process_audio_file, tmp_path)

        _set_cache(cache_key, result)

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info("analyze_success", extra={"file_name": safe_filename, "elapsed_ms": elapsed_ms})
        return result
    except ValueError as e:
        logger.warning("analyze_validation_error", extra={"error": str(e)})
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error("analyze_error", extra={"error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail="Falha interna ao processar o áudio.") from e
    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
                logger.debug("analyze_cleanup", extra={"path": tmp_path})
            except Exception as e:
                logger.exception("analyze_cleanup_error", extra={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SICA_HOST", "0.0.0.0")
    port = int(os.getenv("SICA_PORT", "8002"))
    uvicorn.run("server:app", host=host, port=port, reload=False)
