# OCR Test

Небольшая площадка для проб OCR на Python через EasyOCR.

## Как запустить

1. Создать и активировать виртуальное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Установить зависимости:

```bash
pip install -r requirements.txt
```

3. Положить изображения в папку `input/`.

Поддерживаются: `.jpg`, `.jpeg`, `.png`.

4. Запустить:

```bash
python ocr.py
```

Для каждого изображения будут созданы два `.txt` файла:

- `output/raw/` — исходные строки EasyOCR для отладки
- `output/cleaned/` — упрощенный читаемый текст
