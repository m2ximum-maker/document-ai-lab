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
8. `src/embed.py` строит embeddings и сохраняет их в Chroma.

OCR-результаты перезаписываются при каждом запуске `src/ocr.py`.

## Структура Проекта

```text
.
├── input/                 # входные изображения для OCR
├── output/
│   ├── raw/               # сырой текст EasyOCR
│   ├── cleaned/           # очищенный OCR-текст
│   ├── chunks/            # chunks.jsonl для будущего поиска/RAG
│   └── chroma/            # локальная Chroma vector DB
├── src/
│   ├── __init__.py        # делает src Python-пакетом
│   ├── ocr.py             # OCR pipeline: image → raw/cleaned txt
│   ├── chunk.py           # chunking pipeline: cleaned txt → chunks.jsonl
│   ├── embed.py           # embedding pipeline: chunks.jsonl → Chroma
│   └── schemas.py         # общие типы данных
├── pyproject.toml         # настройки форматирования
├── requirements.txt       # зависимости проекта
├── README.md              # описание проекта
└── .gitignore             # игнор локальных данных и окружения
```

`input/` и `output/` содержат локальные данные и не попадают в git, кроме `.gitkeep` файлов для сохранения структуры папок.

## EXIF Issue

Фото с телефона часто физически лежит боком, а поворот хранится в EXIF. Просмотрщик показывает его правильно, но OCR-библиотека может читать исходные пиксели без поворота. Поэтому перед OCR используется `ImageOps.exif_transpose()`.

## Почему изображения

Основной сценарий — фото документов с телефона, обычно это `.jpg/.jpeg`. PNG тоже поддерживается для скриншотов и тестов. PDF сейчас не поддерживается намеренно, чтобы пайплайн оставался простым.

## Как Запустить

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.ocr
python -m src.chunk
python -m src.embed
```

Первый запуск `src.embed` требует интернет: модель `sentence-transformers/all-MiniLM-L6-v2` скачивается с Hugging Face и затем используется из локального кэша.

## Прогресс

**Готовность MVP: 60%**

```text
██████░░░░ 60%
```

- [x] OCR
- [x] Chunking
- [x] Embeddings
- [ ] Retrieval
- [ ] RAG Q&A

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
- [x] Embeddings MVP
- chunks.jsonl → Chroma
- stale chunks cleanup
- upsert by stable chunk id


## Что Дальше

План развития проекта в сторону document AI / RAG.

- [ ] Add more sample documents
- build small local document archive

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
