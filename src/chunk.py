from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "output" / "cleaned"
OUTPUT_DIR = ROOT_DIR / "output" / "chunks"


def main() -> None: 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Chunking пока не реализован")

    
if __name__ == "__main__":
    main()