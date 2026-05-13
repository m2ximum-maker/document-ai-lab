# OCR Test

Площадка для проб OCR на документах. Цель проекта — быстро проверять, как EasyOCR распознает фото документов, и сравнивать сырой OCR с упрощенной текстовой версией.

## Pipeline

1. Изображения кладутся в `input/`.
2. `src/ocr.py` ищет `.jpg`, `.jpeg`, `.png`.
3. Фото приводится к правильной ориентации через EXIF.
4. EasyOCR распознает текст.
5. OCR-результат сохраняется в две папки:
   - `output/raw/` — исходные строки EasyOCR
   - `output/cleaned/` — немного очищенный текст
6. `src/chunk.py` режет `output/cleaned/*.txt` на чанки.
7. Чанки сохраняются в `output/chunks/chunks.jsonl`.

OCR-результаты перезаписываются при каждом запуске `src/ocr.py`.

## EXIF Issue

Фото с телефона часто физически лежит боком, а поворот хранится в EXIF. Просмотрщик показывает его правильно, но OCR-библиотека может читать исходные пиксели без поворота. Поэтому перед OCR используется `ImageOps.exif_transpose()`.

## Почему изображения

Основной сценарий — фото документов с телефона, обычно это `.jpg/.jpeg`. PNG тоже поддерживается для скриншотов и тестов. PDF сейчас не поддерживается намеренно, чтобы пайплайн оставался простым.

## Как Запустить

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/ocr.py
python src/chunk.py
```

## Что сделано

- [x] OCR MVP
- image → OCR → txt
- EXIF normalization
- batch processing
- JPG-first pipeline
- [x] Chunking MVP
- cleaned txt → chunks.jsonl
- source file metadata
- chunk index metadata


## Что Дальше

План развития проекта в сторону document AI / RAG.

- [ ] Add more sample documents
- build small local document archive

- [ ] Embeddings
- generate embeddings for chunks
- build local vector index

- [ ] Retrieval
- semantic search over document chunks
- top-k similarity search

- [ ] RAG Q&A
- question → retrieval → LLM answer
- include source chunks/files in responses

- [ ] OCR normalization experiments
- compare raw OCR vs LLM-cleaned OCR
- evaluate retrieval quality differences

- [ ] Optional future work
- SQLite archive/catalog
- metadata filtering
- structured extraction (JSON)
- local LLM support
