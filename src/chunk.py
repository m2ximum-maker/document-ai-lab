import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

from src.schemas import Chunk

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "output" / "cleaned"
OUTPUT_CHUNKS_DIR = ROOT_DIR / "output" / "chunks"
CHUNKS_OUTPUT_FILE = OUTPUT_CHUNKS_DIR / "chunks.jsonl"


def main() -> None: 
    OUTPUT_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    text_files = sorted(INPUT_DIR.glob('*.txt'))
    print(f"Найдено файлов для чанкинга: {len(text_files)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    all_chunks: list[Chunk] = []

    for file_path in text_files:
        text = file_path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        
        for index, chunk in enumerate(chunks):
            parsed_chunk: Chunk = {
                "text": chunk,
                "metadata": {
                    "source": file_path.name,
                    "chunk_index": index,
                },
            }
        
            all_chunks.append(parsed_chunk)

    with open(CHUNKS_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Файлов обработано {len(text_files)}")
    print(f"Всего чанков создано {len(all_chunks)}")

if __name__ == "__main__":
    main()
