import json
from json import JSONDecodeError
from pathlib import Path
from typing import cast

import chromadb
from chromadb.api.types import Embeddings
from chromadb.api.types import Metadata
from sentence_transformers import SentenceTransformer

from src.schemas import Chunk

ROOT_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE_PATH = ROOT_DIR / "output" / "chunks" / "chunks.jsonl"
CHROMA_PATH = ROOT_DIR / "output" / "chroma"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "documents"


def validate_chunk(data: object, line_number: int) -> Chunk:
    if not isinstance(data, dict):
        raise ValueError(f"Line {line_number}: chunk must be object")

    text = data.get("text")
    metadata = data.get("metadata")

    if not isinstance(text, str):
        raise ValueError(f"Line {line_number}: text must be string")

    if not isinstance(metadata, dict):
        raise ValueError(f"Line {line_number}: metadata must be object")

    source = metadata.get("source")
    chunk_index = metadata.get("chunk_index")

    if not isinstance(source, str):
        raise ValueError(f"Line {line_number}: metadata.source must be string")

    if not isinstance(chunk_index, int):
        raise ValueError(f"Line {line_number}: metadata.chunk_index must be int")

    return {
        "text": text,
        "metadata": {
            "source": source,
            "chunk_index": chunk_index,
        },
    }


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []

    if not CHUNKS_FILE_PATH.exists():
        print("chunks.jsonl не найден")
        return chunks

    with CHUNKS_FILE_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    data = json.loads(line)
                except JSONDecodeError as error:
                    raise ValueError(f"Line {line_number}: invalid JSON") from error

                chunks.append(validate_chunk(data, line_number))

    return chunks


def main() -> None:
    CHROMA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )
    chunks = load_chunks()

    if not chunks:
        print("No chunks found")
        return

    model = SentenceTransformer(MODEL_NAME)

    texts: list[str] = [chunk["text"] for chunk in chunks]
    metadatas: list[Metadata] = [
        {
            "source": chunk["metadata"]["source"],
            "chunk_index": chunk["metadata"]["chunk_index"],
        }
        for chunk in chunks
    ]
    ids: list[str] = [
        f'{chunk["metadata"]["source"]}_{chunk["metadata"]["chunk_index"]}' for chunk in chunks
    ]

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    current_ids = set(ids)

    existing = collection.get()
    existing_ids = set(existing["ids"])

    stale_ids = list(existing_ids - current_ids)

    if stale_ids:
        collection.delete(ids=stale_ids)

    embeddings = cast(
        Embeddings,
        model.encode(texts).tolist(),
    )

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Saved {len(chunks)} chunks to Chroma")


if __name__ == "__main__":
    main()
