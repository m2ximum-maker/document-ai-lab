import json

import chromadb 
from chromadb.api.types import Metadata
from sentence_transformers import SentenceTransformer
from pathlib import Path

from src.schemas import Chunk

ROOT_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE_PATH = ROOT_DIR / "output" / "chunks" / "chunks.jsonl"
CHROMA_PATH = ROOT_DIR / "output" / "chroma"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "documents"

def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []

    with CHUNKS_FILE_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))
    print(chunks[1])
    return chunks



def main() -> None:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
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
        f'{chunk["metadata"]["source"]}_{chunk["metadata"]["chunk_index"]}'
        for chunk in chunks
    ]

    embeddings = model.encode(texts).tolist() 
    client = chromadb.PersistentClient(path=str(CHROMA_PATH)) 
    collection = client.get_or_create_collection(name=COLLECTION_NAME) 
    collection.add( ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings, ) 
    print(f"Saved {len(chunks)} chunks to Chroma")


if __name__ == "__main__":
    main()
