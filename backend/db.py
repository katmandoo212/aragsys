"""SQLite database layer for query history and metrics."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
import json


DATABASE_PATH = Path("backend/data/aragsys.db")
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class QueryRecord:
    """Query history record."""

    query: str
    pipeline: str
    content: str
    citations: List[str]
    response_time_ms: int
    success: bool
    error_message: Optional[str] = None
    timestamp: Optional[str] = None
    query_id: Optional[int] = None

    def save(self, db_path: Path = DATABASE_PATH) -> int:
        """Save record to database."""
        # Ensure database is initialized
        init_db(db_path)
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO queries
                (query, pipeline, content, citations, response_time_ms,
                 success, error_message, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.query,
                    self.pipeline,
                    self.content,
                    json.dumps(self.citations),
                    self.response_time_ms,
                    self.success,
                    self.error_message,
                    self.timestamp,
                ),
            )
            conn.commit()
            self.query_id = cursor.lastrowid
            return self.query_id

    @classmethod
    def get_recent(cls, db_path: Path = DATABASE_PATH, limit: int = 50) -> List["QueryRecord"]:
        """Get recent query records."""
        # Ensure database is initialized
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM queries
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        records = []
        for row in rows:
            try:
                citations = json.loads(row["citations"])
            except json.JSONDecodeError:
                citations = []

            records.append(
                cls(
                    query_id=row["query_id"],
                    query=row["query"],
                    pipeline=row["pipeline"],
                    content=row["content"],
                    citations=citations,
                    response_time_ms=row["response_time_ms"],
                    success=bool(row["success"]),
                    error_message=row["error_message"],
                    timestamp=row["timestamp"],
                )
            )
        return records

    @classmethod
    def get_metrics(cls, db_path: Path = DATABASE_PATH) -> dict:
        """Get query metrics."""
        # Ensure database is initialized
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Total queries
            cursor.execute("SELECT COUNT(*) FROM queries")
            total = cursor.fetchone()[0]

            # Success rate
            cursor.execute(
                "SELECT COUNT(*) FROM queries WHERE success = 1"
            )
            successful = cursor.fetchone()[0]
            success_rate = (successful / total * 100) if total > 0 else 0

            # Average response time
            cursor.execute(
                "SELECT AVG(response_time_ms) FROM queries WHERE success = 1"
            )
            avg_time = cursor.fetchone()[0] or 0

            # Recent queries (last 24 hours)
            cursor.execute(
                """
                SELECT COUNT(*) FROM queries
                WHERE timestamp > datetime('now', '-1 day')
                """
            )
            recent_24h = cursor.fetchone()[0]

        return {
            "total_queries": total,
            "success_rate": round(success_rate, 2),
            "avg_response_time_ms": round(avg_time, 2),
            "recent_24h": recent_24h,
        }


def init_db(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Initialize database with schema."""
    conn = sqlite3.connect(db_path)

    # Create queries table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            pipeline TEXT NOT NULL,
            content TEXT NOT NULL,
            citations TEXT NOT NULL,
            response_time_ms INTEGER NOT NULL,
            success BOOLEAN NOT NULL,
            error_message TEXT,
            timestamp TEXT NOT NULL
        )
        """
    )

    # Create indexes
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_queries_success ON queries(success)"
    )

    conn.commit()
    return conn


def get_db() -> sqlite3.Connection:
    """Get database connection."""
    if not DATABASE_PATH.exists():
        init_db()
    return sqlite3.connect(DATABASE_PATH)