"""Verify Supabase connectivity (Storage API)."""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.core.supabase_client import get_supabase_client


def main() -> None:
    try:
        client = get_supabase_client()
        buckets = client.storage.list_buckets()

        bucket_names = []
        for bucket in buckets or []:
            name = getattr(bucket, "name", None) or bucket.get("name") if isinstance(bucket, dict) else str(bucket)
            if name:
                bucket_names.append(name)

        print("[OK] Connected to Supabase successfully!")
        print(f"   Storage buckets found: {len(bucket_names)}")
        if bucket_names:
            print(f"   Buckets: {', '.join(bucket_names)}")
        else:
            print("   No storage buckets yet (connection is still valid).")
    except Exception as exc:
        print("[FAIL] Failed to connect to Supabase:")
        print(f"   {type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
