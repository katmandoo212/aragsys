"""Tests for database layer."""

import sqlite3
import tempfile
from pathlib import Path

from backend.db import get_db, init_db, QueryRecord


def test_init_db_creates_tables():
    """Database initialization creates required tables."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = None
    try:
        conn = init_db(db_path)
        cursor = conn.cursor()

        # Check queries table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='queries'"
        )
        assert cursor.fetchone() is not None
    finally:
        if conn is not None:
            conn.close()
        # Give SQLite time to release locks on Windows
        import gc
        import time
        gc.collect()
        time.sleep(0.2)
        db_path.unlink()


def test_query_record_save_and_retrieve():
    """QueryRecord can be saved and retrieved from database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = None
    try:
        conn = init_db(db_path)
        conn.close()

        # Save a query record
        record = QueryRecord(
            query="test query",
            pipeline="naive_flow",
            content="answer content",
            citations=["doc1"],
            response_time_ms=1000,
            success=True
        )
        record.save(db_path)

        # Retrieve queries
        queries = QueryRecord.get_recent(db_path, limit=10)
        assert len(queries) == 1
        assert queries[0].query == "test query"
        assert queries[0].success is True

    finally:
        # Give SQLite time to release locks on Windows
        import gc
        import time
        gc.collect()
        time.sleep(0.2)
        db_path.unlink()