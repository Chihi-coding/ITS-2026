"""Supabase client initialization."""

from functools import lru_cache

from supabase import Client, create_client

from app.core.config import SUPABASE_KEY, SUPABASE_URL


def _validate_config() -> None:
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


@lru_cache
def get_supabase_client() -> Client:
    """Return a cached Supabase client instance."""
    _validate_config()
    return create_client(SUPABASE_URL, SUPABASE_KEY)
