import contextlib
import io
import textwrap
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")


def find_images() -> list[Path]:
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

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
    text = " ".join(line.strip() for line in lines if line.strip())
    return textwrap.fill(text, width=100)


image_paths = find_images()
print(f"Найдено изображений: {len(image_paths)}")

print("Запускаю EasyOCR...")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    reader = easyocr.Reader(["ru", "en"], gpu=False)

for image_path in image_paths:
    result_path = OUTPUT_DIR / f"{image_path.stem}.txt"
    if result_path.exists():
        print(f"Пропускаю, уже есть результат: {result_path}")
        continue

    print(f"Обрабатываю: {image_path}")
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        lines = [str(line) for line in reader.readtext(np.array(image), detail=0)]

    text = clean_text(lines)
    result_path.write_text(f"Источник: {image_path.name}\n\n{text}\n", encoding="utf-8")
    print(f"Сохранено: {result_path}")

print("\nГотово")
