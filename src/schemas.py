from typing import TypedDict


class ChunkMetadata(TypedDict):
    source: str
    chunk_index: int


class Chunk(TypedDict):
    text: str
    metadata: ChunkMetadata
