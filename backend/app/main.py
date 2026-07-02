"""FastAPI application entrypoint."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.violations import router as violations_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Parking System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(violations_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/supabase")
def supabase_health_check() -> dict[str, object]:
    """Quick Supabase connectivity check for troubleshooting 500 errors."""
    from app.core.config import SUPABASE_KEY, SUPABASE_URL

    if not SUPABASE_URL or not SUPABASE_KEY:
        return {
            "ok": False,
            "error": "Missing SUPABASE_URL or SUPABASE_KEY in .env",
        }

    try:
        from app.core.supabase_client import get_supabase_client

        client = get_supabase_client()
        buckets = client.storage.list_buckets()
        bucket_names: list[str] = []
        for bucket in buckets or []:
            name = getattr(bucket, "name", None)
            if not name and isinstance(bucket, dict):
                name = bucket.get("name")
            if name:
                bucket_names.append(str(name))
        violations = client.table("violations").select("id").limit(1).execute()
        return {
            "ok": True,
            "buckets": bucket_names,
            "violations_table": "ok" if violations.data is not None else "missing",
            "violation_images_bucket": "violation-images" in bucket_names,
        }
    except Exception as exc:
        logger.exception("Supabase health check failed")
        return {"ok": False, "error": str(exc)}


@app.on_event("startup")
def on_startup() -> None:
    logger.info("Smart Parking API started")
