import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from typing import cast

from chromadb.api.types import Embedding
from chromadb.api.types import Metadata

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT_DIR / "output" / "chroma"


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

COLLECTION_NAME = "documents"
SEARCH_TOP_K = 10
EXPAND_TOP_N = 5
NEIGHBOR_WINDOW = 1


def metadata_key(metadata: Metadata) -> tuple[str, int] | None:
    source = metadata.get("source")
    chunk_index = metadata.get("chunk_index")

    if not isinstance(source, str) or not isinstance(chunk_index, int):
        return None

    return source, chunk_index


def keyword_score(query: str, document: str) -> int:
    query_words = set(query.lower().split())
    document_lower = document.lower()

    return sum(1 for word in query_words if word in document_lower)


def search_chunks(query: str, top_chunks: int = SEARCH_TOP_K) -> None:
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

    if not results["documents"] or not results["metadatas"] or not results["distances"]:
        print("Ничего не найдено или база пуста")
        return

    found_documents = results["documents"][0]
    found_metadatas = results["metadatas"][0]
    found_distances = results["distances"][0]

    hits: list[tuple[str, int, str, float]] = []

    for document, metadata, distance in zip(
        found_documents,
        found_metadatas,
        found_distances,
    ):
        key = metadata_key(metadata)

        if key is None:
            continue

        source, chunk_index = key
        hits.append((source, chunk_index, document, distance))

    if not hits:
        print("Ничего не найдено или база пуста")
        return

    distance_by_key: dict[tuple[str, int], float] = {}
    wanted_by_source: dict[str, set[int]] = {}
    source_order: list[str] = []

    for source, chunk_index, _, distance in hits[:EXPAND_TOP_N]:
        distance_by_key[(source, chunk_index)] = distance

        if source not in wanted_by_source:
            wanted_by_source[source] = set()
            source_order.append(source)

        for neighbor_index in range(
            chunk_index - NEIGHBOR_WINDOW,
            chunk_index + NEIGHBOR_WINDOW + 1,
        ):
            if neighbor_index >= 0:
                wanted_by_source[source].add(neighbor_index)

    chunks_by_source: dict[str, list[tuple[int, str, float | None]]] = {}

    for source in source_order:
        source_results = collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        documents = source_results["documents"]
        metadatas = source_results["metadatas"]

        if documents is None or metadatas is None:
            continue

        wanted_indexes = wanted_by_source[source]
        chunks: list[tuple[int, str, float | None]] = []

        for document, metadata in zip(documents, metadatas):
            key = metadata_key(metadata)

            if key is None:
                continue

            _, chunk_index = key

            if chunk_index in wanted_indexes:
                chunks.append(
                    (
                        chunk_index,
                        document,
                        distance_by_key.get((source, chunk_index)),
                    )
                )

        chunks_by_source[source] = sorted(chunks)

    source_order = [source for source in source_order if chunks_by_source[source]]

    source_order.sort(
        key=lambda source: (
            -max(keyword_score(query, document) for _, document, _ in chunks_by_source[source]),
            min(
                distance
                for _, _, distance in chunks_by_source[source]
                if distance is not None
            ),
        )
    )

    context: list[tuple[str, int, str, float | None]] = [
        (source, chunk_index, document, distance)
        for source in source_order
        for chunk_index, document, distance in chunks_by_source[source]
    ]

    if not context:
        print("Ничего не найдено или база пуста")
        return

    print(f"Найдено через vector search: {len(hits)}")
    print(f"Расширено соседями top hits: {min(len(hits), EXPAND_TOP_N)}")
    print(f"Итоговый context chunks: {len(context)}")

    for i, (source, chunk_index, document, distance) in enumerate(
        context,
        start=1,
    ):
        distance_text = distance if distance is not None else "neighbor"
        print(f"\n[{i}] distance={distance_text}")
        print(f"source={source}")
        print(f"chunk_index={chunk_index}")
        print(document)


def main() -> None:
    if len(sys.argv) < 2:
        print('Использование: python -m src.search "ваш вопрос"')
        return

    query = " ".join(sys.argv[1:])
    search_chunks(query)


if __name__ == "__main__":
    main()
