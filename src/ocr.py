import contextlib
import io
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"
RAW_OUTPUT_DIR = OUTPUT_DIR / "raw"
CLEANED_OUTPUT_DIR = OUTPUT_DIR / "cleaned"


def find_images() -> list[Path]:
    INPUT_DIR.mkdir(exist_ok=True)
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise FileNotFoundError(f"Положи изображения в папку {INPUT_DIR}: {extensions}")
    return sorted(images)


def clean_text(lines: list[str]) -> str:
    cleaned = [line.strip() for line in lines if line.strip()]

    result = []
    current = []

    for line in cleaned:
        current.append(line)

        if line.endswith((".", ":")):
            result.append(" ".join(current))
            current = []

    if current:
        result.append(" ".join(current))

    return "\n".join(result)


image_paths = find_images()
print(f"Найдено изображений: {len(image_paths)}")

print("Запускаю EasyOCR...")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    reader = easyocr.Reader(["ru", "en"], gpu=False)

for image_path in image_paths:
    raw_path = RAW_OUTPUT_DIR / f"{image_path.stem}.txt"
    cleaned_path = CLEANED_OUTPUT_DIR / f"{image_path.stem}.txt"

    if raw_path.exists() and cleaned_path.exists():
        print(f"Пропускаю, уже есть результат: {cleaned_path}")
        continue

    print(f"Обрабатываю: {image_path}")
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        lines = [str(line) for line in reader.readtext(np.array(image), detail=0)]

    raw_text = "\n".join(lines)
    cleaned_text = clean_text(lines)

    raw_path.write_text(f"Источник: {image_path.name}\n\n{raw_text}\n", encoding="utf-8")
    cleaned_path.write_text(f"Источник: {image_path.name}\n\n{cleaned_text}\n", encoding="utf-8")

    print(f"Сохранено raw: {raw_path}")
    print(f"Сохранено cleaned: {cleaned_path}")

print("\nГотово")
