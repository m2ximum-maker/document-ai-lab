from __future__ import annotations

from pathlib import Path

from src.schemas import DocumentMetadata

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
CLEANED_OUTPUT_DIR = OUTPUT_DIR / "cleaned"
METADATA_OUTPUT_DIR = OUTPUT_DIR / "metadata"

METADATA_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "source",
        "owner",
        "doc_type",
        "doc_date",
        "doctor",
        "specialty",
        "clinic",
        "summary",
        "confidence",
        "warnings",
    ],
    "properties": {
        "source": {"type": "string"},
        "owner": {"type": ["string", "null"]},
        "doc_type": {"type": ["string", "null"]},
        "doc_date": {"type": ["string", "null"]},
        "doctor": {"type": ["string", "null"]},
        "specialty": {"type": ["string", "null"]},
        "clinic": {"type": ["string", "null"]},
        "summary": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}


def build_empty_metadata(source: str) -> DocumentMetadata:
    return {
        "source": source,
        "owner": None,
        "doc_type": None,
        "doc_date": None,
        "doctor": None,
        "specialty": None,
        "clinic": None,
        "summary": None,
        "confidence": 0.0,
        "warnings": [],
    }


def find_cleaned_documents() -> list[Path]:
    if not CLEANED_OUTPUT_DIR.exists():
        return []

    return sorted(CLEANED_OUTPUT_DIR.glob("*.txt"))


def main() -> None:
    METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = find_cleaned_documents()

    print(f"Найдено документов: {len(documents)}")
    print(f"Metadata output: {METADATA_OUTPUT_DIR.relative_to(ROOT_DIR)}")

    for path in documents:
        metadata = build_empty_metadata(path.name)
        print(f"- {metadata['source']}")


if __name__ == "__main__":
    main()
