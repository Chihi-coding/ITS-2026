"""Load environment variables from the project root .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = ROOT_DIR / "backend"

for env_path in (ROOT_DIR / ".env", BACKEND_DIR / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=True)


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


def normalize_supabase_url(url: str | None) -> str | None:
    """Strip REST suffixes so create_client gets the project base URL."""
    if not url:
        return None
    cleaned = url.rstrip("/")
    for suffix in ("/rest/v1", "/auth/v1", "/storage/v1"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned.rstrip("/")


SUPABASE_URL = normalize_supabase_url(get_env("SUPABASE_URL"))
SUPABASE_KEY = get_env("SUPABASE_KEY") or get_env("SUPABASE_SECRET_KEY")
DATABASE_URL = get_env("DATABASE_URL")
TELEGRAM_BOT_TOKEN = get_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_env("TELEGRAM_CHAT_ID")
DEBUG = get_env("DEBUG", "false").lower() in {"1", "true", "yes", "on"}
