import sqlite3

from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
CLEANED_OUTPUT_DIR = OUTPUT_DIR / "cleaned"
CATALOG_DIR = OUTPUT_DIR / "catalog"
DB_PATH = CATALOG_DIR / "catalog.db"


def import_documents(connection: sqlite3.Connection) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for path in sorted(CLEANED_OUTPUT_DIR.glob("*.txt")):
        source = path.name
        filename = path.name

        connection.execute(
            """
            INSERT INTO documents (
                source,
                filename,
                owner,
                doc_type,
                doc_date,
                created_at
            )
            VALUES (?, ?, NULL, NULL, NULL, ?)
            ON CONFLICT(source) DO UPDATE SET
                filename = excluded.filename
            """,
            (source, filename, now),
        )

    connection.commit()


def init_db() -> sqlite3.Connection:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            owner TEXT NULL,
            doc_type TEXT NULL,
            doc_date TEXT NULL,
            created_at TEXT NOT NULL
        )
        """)
    connection.commit()
    return connection


def main() -> None:
    connection = init_db()
    db_display_path = DB_PATH.relative_to(ROOT_DIR)

    try:
        import_documents(connection)
        count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

        print(f"Catalog updated: {count} documents")
    finally:
        connection.close()

    print(f"Catalog path: {db_display_path}")


if __name__ == "__main__":
    main()
