# Smart Medical Archive

Личный архив медицинских документов для всей семьи.

Проект помогает превращать фото и сканы документов в текст, искать по ним, задавать вопросы и постепенно собирать структурированную медицинскую историю с указанием источников.

Коротко:

> Личный ChatGPT по медицинским документам семьи.

Важно: это не медицинский советчик, не диагностическая система и не замена врачу. Проект нужен, чтобы помнить, находить, группировать и цитировать уже существующие документы.

## Куда Идём

Сейчас проект вырос из OCR/RAG-песочницы в Smart Medical Archive.

Цель: не просто искать по файлам вроде `IMG_5371.txt`, а находить информацию по медицинской истории:

- ЖКТ
- ЛОР
- кровь
- спина
- аллергия
- лекарства
- ветдокументы

Пользователь в будущем должен иметь возможность спрашивать:

```text
Что мне говорил ЛОР?
Какие лекарства мне назначали от желудка?
Когда последний раз сдавал кровь?
Что было по щитовидке?
К каким врачам я ходил с марта по май 2026 года?
Что назначали питомцу?
```

Подробное описание направления проекта: [docs/PROJECT_DIRECTION.md](docs/PROJECT_DIRECTION.md).

Текущий прогресс и ближайшие шаги: [docs/PROGRESS.md](docs/PROGRESS.md).

## Текущий Pipeline

```text
input images
↓
OCR
↓
output/cleaned/*.txt
↓
src/chunk.py
↓
output/chunks/chunks.jsonl
↓
src/embed.py / Chroma
↓
src/search.py / src.ask.py
```

Подробно:

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
9. `src/search.py` ищет релевантные чанки и собирает context для RAG.
10. `src/ask.py` собирает prompt из context + question, отправляет его в OpenAI API и печатает ответ с источниками.

OCR-результаты перезаписываются при каждом запуске `src/ocr.py`.

## SQLite Catalog

Следующий архитектурный слой — SQLite-каталог рядом с текущим pipeline.

SQLite нужен не вместо Chroma. Их роли разные:

- Chroma хранит semantic index по чанкам и помогает искать похожий текст.
- SQLite хранит документы и структурированные metadata.

Будущие metadata после OCR:

- владелец документа
- дата документа
- тип документа
- врач
- специальность
- клиника
- статус и уверенность извлечения

На первом шаге каталог добавляется аккуратно и отдельно от текущего поиска: без изменения `chunk.py`, `search.py`, `ask.py`, `chunks.jsonl` и без миграции Chroma.

## Структура Проекта

```text
.
├── docs/
│   ├── PROJECT_DIRECTION.md # направление проекта и роль SQLite/metadata
│   └── PROGRESS.md          # прогресс, ближайшие шаги и guardrails
├── input/                   # входные изображения для OCR
├── output/
│   ├── raw/                 # сырой текст EasyOCR
│   ├── cleaned/             # очищенный OCR-текст
│   ├── chunks/              # chunks.jsonl для поиска/RAG
│   └── chroma/              # локальная Chroma vector DB
├── eval/
│   └── eval_queries.example.json # пример retrieval eval cases
├── tests/
│   └── test_search_rrf.py   # unit tests для hybrid RRF merge
├── src/
│   ├── __init__.py          # делает src Python-пакетом
│   ├── ocr.py               # OCR pipeline: image -> raw/cleaned txt
│   ├── chunk.py             # chunking pipeline: cleaned txt -> chunks.jsonl
│   ├── embed.py             # embedding pipeline: chunks.jsonl -> Chroma
│   ├── eval.py              # простой retrieval smoke test
│   ├── search.py            # retrieval MVP: query -> context chunks
│   ├── ask.py               # RAG Q&A MVP: question -> retrieval -> LLM answer
│   └── schemas.py           # общие типы данных
├── pyproject.toml           # настройки форматирования
├── requirements.txt         # зависимости проекта
├── README.md                # точка входа в проект
└── .gitignore               # игнор локальных данных и окружения
```

`input/` и `output/` содержат локальные данные и не попадают в git, кроме `.gitkeep` файлов для сохранения структуры папок.

## Как Запустить

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.ocr
python -m src.chunk
python -m src.embed
python -m src.search "ваш вопрос для получения нужного чанка"
python -m src.ask "ваш вопрос к llm"
```

Первый запуск `src.embed` требует интернет: модель `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` скачивается с Hugging Face и затем используется из локального кэша.

Для полностью offline-запуска после загрузки модели можно использовать:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m src.search "ваш вопрос"
```

Для RAG Q&A нужен OpenAI API key и имя модели в `.env`:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=your_model_name_here
```

`.env` игнорируется git и предназначен только для локальных секретов и настроек.

## Retrieval MVP

`src/search.py` сейчас делает MVP hybrid-поиск по локальной Chroma collection и `chunks.jsonl`:

- vector search по embeddings, `top_k=10`
- BM25 keyword search по `output/chunks/chunks.jsonl`, `top_k=10`
- merge vector hits + BM25 hits через RRF
- расширение соседними chunks для top-5 результатов, `neighbor_window=1`
- дедупликация chunks
- сохранение порядка источников из hybrid ranking
- сохранение порядка chunks внутри одного source по `chunk_index`
- основной CLI печатает metadata и полный текст context chunks
- отдельный debug-режим BM25 keyword search через `--bm25`

Это повышает шанс, что LLM получит не только найденный chunk, но и соседний контекст из того же документа. BM25 помогает точным словам, датам, фамилиям и медицинским терминам; vector search помогает смысловым совпадениям. RRF объединяет обе выдачи по рангу, не сравнивая напрямую vector distance и BM25 score.

BM25 debug-запуск:

```bash
python -m src.search --bm25 "ваш вопрос"
```

## Retrieval Eval

`src/eval.py` — простой smoke/quality test для retrieval. Он читает eval cases, запускает `retrieve_context(query)` и проверяет только одно: попал ли `expected_source` в найденный context.

Формат локального `eval/eval_queries.json`:

```json
[
  {
    "query": "Когда была последнее посещение терапевта?",
    "expected_source": "IMG_1.txt"
  }
]
```

Запуск:

```bash
python -m src.eval eval/eval_queries.json
```

`eval/eval_queries.json` игнорируется git, потому что может содержать личные данные. Для коммита есть обезличенный пример: `eval/eval_queries.example.json`.

## RAG Q&A

`src/ask.py` — минимальный RAG поверх существующего retrieval:

- получает вопрос из CLI
- ищет context через `retrieve_context(question)`
- превращает найденные `ContextChunk` в текстовый context
- собирает prompt из инструкции, context и question
- отправляет prompt в OpenAI API
- печатает короткие CLI-статусы для долгих этапов
- стримит ответ модели по мере генерации
- печатает источники из использованного context, сгруппированные по файлам и chunk indexes

Запуск:

```bash
python -m src.ask "Когда была эндоскопия желудка?"
```

Пример CLI-вывода:

```text
Ищу контекст...
Найдено context chunks: 10
Генерирую ответ...
Эндоскопия желудка была ...
Источники:
- IMG_5371 2.txt: chunks 0, 1, 2
```

Источники формируются кодом из metadata найденных chunks, а не генерируются LLM. Это делает вывод стабильнее, но список показывает context, который был передан модели, а не точное доказательство того, какой именно chunk содержал конкретную фразу ответа.

Если retrieval не вернул context, CLI печатает `Контекст не найден` и не вызывает OpenAI API.

## EXIF Issue

Фото с телефона часто физически лежит боком, а поворот хранится в EXIF. Просмотрщик показывает его правильно, но OCR-библиотека может читать исходные пиксели без поворота. Поэтому перед OCR используется `ImageOps.exif_transpose()`.

## Почему Изображения

Основной сценарий — фото документов с телефона, обычно это `.jpg/.jpeg`. PNG тоже поддерживается для скриншотов и тестов. PDF сейчас не поддерживается намеренно, чтобы пайплайн оставался простым.

## Known Rough Edges

Что нужно будет улучшить после MVP:

- hybrid search может поднять OCR-мусор или нерелевантный keyword match выше смыслового результата
- top-1 выдаче пока нельзя полностью доверять
- neighbor expansion повышает recall, но может подтягивать соседей от нерелевантных hits
- RRF merge помогает объединять vector/BM25, но это не полноценный reranker
- короткие однословные запросы и запросы по именам/кличкам всё ещё требуют tuning
- синонимы и аббревиатуры могут промахиваться из-за OCR-шума
- OCR-шум влияет и на embeddings, и на keyword matching
- итоговый context пока может содержать лишние источники; позже нужен лимит/threshold
- RAG-ответ зависит от качества OCR: модель может аккуратно восстанавливать очевидный смысл, но не должна выдумывать факты
- `src.ask` выводит источники использованного context, но пока не определяет точный chunk-доказательство для каждого факта ответа
- для offline-режима надёжнее использовать `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, а не `local_files_only=True` в коде
