import json

import chromadb
from chromadb.api.types import Metadata
from sentence_transformers import SentenceTransformer
from pathlib import Path

from src.schemas import Chunk

from typing import cast

from chromadb.api.types import Embeddings

ROOT_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE_PATH = ROOT_DIR / "output" / "chunks" / "chunks.jsonl"
CHROMA_PATH = ROOT_DIR / "output" / "chroma"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "documents"


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []

    if not CHUNKS_FILE_PATH.exists():
        print("chunks.jsonl not found")
        return chunks

    with CHUNKS_FILE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

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
    ids: list[str] = [f'{chunk["metadata"]["source"]}_{chunk["metadata"]["chunk_index"]}' for chunk in chunks]

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
