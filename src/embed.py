import json

from pathlib import Path

from src.schemas import Chunk

ROOT_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE_PATH = ROOT_DIR / "output" / "chunks" / "chunks.jsonl"
CHROMA_PATH = ROOT_DIR / "output" / "chroma"


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
    load_chunks()

if __name__ == "__main__":
    main()
