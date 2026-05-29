import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import chromadb
from sentence_transformers import SentenceTransformer

from chromadb.api.models.Collection import Collection
from chromadb.api.types import Embedding
from chromadb.api.types import Metadata
from chromadb.api.types import QueryResult

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT_DIR / "output" / "chroma"


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

COLLECTION_NAME = "documents"

SEARCH_TOP_K = 10
EXPAND_TOP_N = 5
NEIGHBOR_WINDOW = 1

# tuple[source, chunk_index]
ChunkKey = tuple[str, int]


@dataclass(frozen=True)
class Hit:
    source: str
    chunk_index: int
    document: str
    distance: float


@dataclass(frozen=True)
class ExpandedChunk:
    chunk_index: int
    document: str
    distance: float | None


@dataclass(frozen=True)
class ContextChunk:
    source: str
    chunk_index: int
    document: str
    distance: float | None


@dataclass(frozen=True)
class NeighborExpansionPlan:
    distance_by_key: dict[ChunkKey, float]
    wanted_by_source: dict[str, set[int]]
    source_order: list[str]


def metadata_key(metadata: Metadata) -> tuple[str, int] | None:
    # Достаём из metadata стабильный ключ чанка: исходный файл + номер чанка.
    # Если Chroma вернула metadata неожиданного формата, такой результат пропускаем.
    source = metadata.get("source")
    chunk_index = metadata.get("chunk_index")

    if not isinstance(source, str) or not isinstance(chunk_index, int):
        return None

    return source, chunk_index


def keyword_score(query: str, document: str) -> int:
    # Простейшая keyword-оценка: считаем, сколько слов из запроса встречается в тексте.
    # Она нужна только как дополнительный сигнал при сортировке документов.
    query_words = set(query.lower().split())
    document_lower = document.lower()

    return sum(1 for word in query_words if word in document_lower)


def get_collection() -> Collection:
    # Подключаемся к локальной Chroma-базе с уже сохранёнными chunks.
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(name=COLLECTION_NAME)


def embed_query(query: str, model: SentenceTransformer) -> Embedding:
    # Превращаем пользовательский вопрос в embedding того же формата,
    # что и embeddings сохранённых чанков.
    return cast(
        Embedding,
        model.encode(query, normalize_embeddings=True).tolist(),
    )


def query_nearest_chunks(
    collection: Collection,
    query_embedding: Embedding,
    top_chunks: int,
) -> QueryResult:
    # Ищем ближайшие чанки по vector similarity.
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_chunks,
    )


def normalize_hits(results: QueryResult) -> list[Hit]:
    # Chroma возвращает списки списков: внешний уровень нужен для batch-запросов.
    # Здесь запрос один, поэтому дальше работаем с нулевым элементом.
    if not results["documents"] or not results["metadatas"] or not results["distances"]:
        return []

    found_documents = results["documents"][0]
    found_metadatas = results["metadatas"][0]
    found_distances = results["distances"][0]

    # Нормализуем сырые результаты Chroma в удобный список hits.
    # Заодно отбрасываем элементы с битой или неполной metadata.
    hits: list[Hit] = []

    for document, metadata, distance in zip(
        found_documents,
        found_metadatas,
        found_distances,
    ):
        key = metadata_key(metadata)

        if key is None:
            continue

        source, chunk_index = key
        hits.append(
            Hit(
                source=source,
                chunk_index=chunk_index,
                document=document,
                distance=distance,
            )
        )

    return hits


def plan_neighbor_expansion(hits: list[Hit]) -> NeighborExpansionPlan:
    # distance хранится только для чанков, которые реально нашёл vector search.
    # У соседних чанков distance будет None, потому что они добавляются как контекст.
    distance_by_key: dict[ChunkKey, float] = {}
    wanted_by_source: dict[str, set[int]] = {}
    source_order: list[str] = []

    # Для лучших найденных чанков собираем сам chunk и соседние,
    # чтобы восстановить контекст, разрезанный chunking'ом.
    # Группируем нужные chunk indexes по source-документу.
    for hit in hits[:EXPAND_TOP_N]:
        distance_by_key[(hit.source, hit.chunk_index)] = hit.distance

        if hit.source not in wanted_by_source:
            wanted_by_source[hit.source] = set()
            source_order.append(hit.source)

        for neighbor_index in range(
            hit.chunk_index - NEIGHBOR_WINDOW,
            hit.chunk_index + NEIGHBOR_WINDOW + 1,
        ):
            if neighbor_index >= 0:
                wanted_by_source[hit.source].add(neighbor_index)

    return NeighborExpansionPlan(
        distance_by_key=distance_by_key,
        wanted_by_source=wanted_by_source,
        source_order=source_order,
    )


def load_expanded_chunks(
    collection: Collection,
    expansion_plan: NeighborExpansionPlan,
) -> dict[str, list[ExpandedChunk]]:
    # Загружаем chunks для найденных документов
    # и оставляем только нужные chunk indexes:
    # найденные retrieval'ом + соседние chunks.
    chunks_by_source: dict[str, list[ExpandedChunk]] = {}

    for source in expansion_plan.source_order:
        source_results = collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        documents = source_results["documents"]
        metadatas = source_results["metadatas"]

        if documents is None or metadatas is None:
            continue

        wanted_indexes = expansion_plan.wanted_by_source[source]
        chunks: list[ExpandedChunk] = []

        for document, metadata in zip(documents, metadatas):
            key = metadata_key(metadata)

            if key is None:
                continue

            _, chunk_index = key

            if chunk_index in wanted_indexes:
                chunks.append(
                    ExpandedChunk(
                        chunk_index=chunk_index,
                        document=document,
                        distance=expansion_plan.distance_by_key.get((source, chunk_index)),
                    )
                )

        chunks_by_source[source] = sorted(chunks, key=lambda chunk: chunk.chunk_index)

    return chunks_by_source


def sort_sources_by_hybrid_score(
    query: str,
    chunks_by_source: dict[str, list[ExpandedChunk]],
    source_order: list[str],
) -> list[str]:
    # Пересортировываем документы по простой hybrid-логике:
    # сначала по keyword совпадениям, затем по vector distance.
    sorted_sources = [source for source in source_order if chunks_by_source.get(source)]

    sorted_sources.sort(
        key=lambda source: (
            -max(keyword_score(query, chunk.document) for chunk in chunks_by_source[source]),
            min(chunk.distance for chunk in chunks_by_source[source] if chunk.distance is not None),
        )
    )

    return sorted_sources


def build_context(
    source_order: list[str],
    chunks_by_source: dict[str, list[ExpandedChunk]],
) -> list[ContextChunk]:
    # Разворачиваем сгруппированные по source чанки обратно в один итоговый context.
    return [
        ContextChunk(
            source=source,
            chunk_index=chunk.chunk_index,
            document=chunk.document,
            distance=chunk.distance,
        )
        for source in source_order
        for chunk in chunks_by_source[source]
    ]


def print_context(hits: list[Hit], context: list[ContextChunk]) -> None:
    print(f"Найдено через vector search: {len(hits)}")
    print(f"Расширено соседями top hits: {min(len(hits), EXPAND_TOP_N)}")
    print(f"Итоговый context chunks: {len(context)}")

    # Печатаем итоговый контекст в порядке, который будет удобно скопировать
    # или передать дальше в LLM/RAG-пайплайн.
    for i, chunk in enumerate(
        context,
        start=1,
    ):
        distance_text = chunk.distance if chunk.distance is not None else "neighbor"
        print(f"\n[{i}] distance={distance_text}")
        print(f"source={chunk.source}")
        print(f"chunk_index={chunk.chunk_index}")
        print(chunk.document[:300])


def search_chunks(query: str, top_chunks: int = SEARCH_TOP_K) -> None:
    collection = get_collection()

    # Загружаем embedding-модель только для пользовательского запроса.
    model = SentenceTransformer(MODEL_NAME)

    query_embedding = embed_query(query, model)
    results = query_nearest_chunks(collection, query_embedding, top_chunks)
    hits = normalize_hits(results)

    if not hits:
        print("Ничего не найдено или база пуста")
        return

    # Какие source/chunk_index нужно дозагрузить (соседние чанки)
    # и какие distance были у чанков, найденных самим vector search.
    expansion_plan = plan_neighbor_expansion(hits)

    # Чанки, сгруппированные по source: найденные chunks + их соседи.
    chunks_by_source = load_expanded_chunks(collection, expansion_plan)

    # Source-документы, пересортированные по hybrid-оценке:
    # keyword score, затем vector distance.
    sorted_sources = sort_sources_by_hybrid_score(
        query,
        chunks_by_source,
        expansion_plan.source_order,
    )

    # Финальный плоский список chunks в порядке, в котором их печатаем
    # или передаём дальше в RAG-контекст.
    context = build_context(sorted_sources, chunks_by_source)

    if not context:
        print("Ничего не найдено или база пуста")
        return

    print_context(hits, context)


def main() -> None:
    if len(sys.argv) < 2:
        print('Использование: python -m src.search "ваш вопрос"')
        return

    query = " ".join(sys.argv[1:])
    search_chunks(query)


if __name__ == "__main__":
    main()
