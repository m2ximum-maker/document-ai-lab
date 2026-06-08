from typing import Optional, TypedDict


class ChunkMetadata(TypedDict):
    source: str
    chunk_index: int


class Chunk(TypedDict):
    text: str
    metadata: ChunkMetadata


class DocumentMetadata(TypedDict):
    source: str
    owner: Optional[str]
    doc_type: Optional[str]
    doc_date: Optional[str]
    doctor: Optional[str]
    specialty: Optional[str]
    clinic: Optional[str]
    summary: Optional[str]
    confidence: float
    warnings: list[str]
