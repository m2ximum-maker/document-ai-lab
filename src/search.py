import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from typing import cast

from chromadb.api.types import Embedding

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT_DIR / "output" / "chroma"


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

COLLECTION_NAME = "documents"


def search_chunks(query: str, top_chunks: int = 5) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(MODEL_NAME)

    query_embedding = cast(
        Embedding,
        model.encode(query, normalize_embeddings=True).tolist(),
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_chunks,
    )

    if results["documents"] and results["metadatas"] and results["distances"]:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
    else:
        print("Ничего не найдено или база пуста")

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        print(f"\n[{i}] distance={distance}")
        print(f"source={metadata.get('source')}")
        print(f"chunk_index={metadata.get('chunk_index')}")
        print(document)

    print(f"Синхронизирую Chroma collection: {COLLECTION_NAME}")


def main() -> None:
    if len(sys.argv) < 2:
        print('Использование: python -m src.search "ваш вопрос"')

    query = " ".join(sys.argv[1:])
    search_chunks(query)


if __name__ == "__main__":
    main()
