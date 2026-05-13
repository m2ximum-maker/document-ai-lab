from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "output" / "cleaned"
OUTPUT_DIR = ROOT_DIR / "output" / "chunks"
CHUNKS_OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"


def main() -> None: 
    text_files = list(INPUT_DIR.glob('*.txt'))
    print(text_files)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    for file_path in text_files:
        text = file_path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        print(f"{file_path.name}: {len(chunks)} chunks")
    
if __name__ == "__main__":
    main()