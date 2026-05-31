# OCR Test

Площадка для проб OCR, document retrieval и RAG Q&A. Цель проекта — быстро проверять, как EasyOCR распознает фото документов, превращать результат в chunks/embeddings, искать релевантный контекст и отвечать на вопросы по найденным фрагментам.

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
9. `src/search.py` ищет релевантные чанки и собирает context для RAG.
10. `src/ask.py` собирает prompt из context + question, отправляет его в OpenAI API и печатает ответ с источниками.

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
├── eval/
│   └── eval_queries.example.json # пример retrieval eval cases
├── src/
│   ├── __init__.py        # делает src Python-пакетом
│   ├── ocr.py             # OCR pipeline: image → raw/cleaned txt
│   ├── chunk.py           # chunking pipeline: cleaned txt → chunks.jsonl
│   ├── embed.py           # embedding pipeline: chunks.jsonl → Chroma
│   ├── eval.py            # простой retrieval smoke test
│   ├── search.py          # retrieval MVP: query → context chunks
│   ├── ask.py             # RAG Q&A MVP: question → retrieval → LLM answer
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

`src/search.py` сейчас делает MVP-поиск по локальной Chroma collection:

- vector search по embeddings, `top_k=10`
- расширение соседними chunks для top-5 результатов, `neighbor_window=1`
- дедупликация chunks
- сортировка источников с простым keyword boost по словам запроса
- сохранение порядка chunks внутри одного source по `chunk_index`
- CLI печатает metadata и preview текста чанка до 300 символов

Это повышает шанс, что LLM получит не только найденный chunk, но и соседний контекст из того же документа. Ограничение текущего подхода: это всё ещё не полноценный hybrid/BM25 search, поэтому OCR-шум и короткие запросы могут давать лишние chunks.

## Retrieval Eval

`src/eval.py` — простой smoke/quality test для retrieval. Он читает eval cases, запускает `retrieve_context(query)` и проверяет только одно: попал ли `expected_source` в найденный context. Качество ответа LLM, точность chunk index и наличие конкретных фраз внутри текста пока не оцениваются.

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

Для MVP нормальна ситуация, когда часть quality cases падает: это показывает слабые места retrieval на OCR-шуме, коротких запросах и синонимах. Такой eval нужен как baseline, чтобы видеть, стали ли изменения лучше или хуже.

## RAG Q&A

`src/ask.py` — минимальный RAG поверх существующего retrieval:

- получает вопрос из CLI
- ищет context через `retrieve_context(question)`
- превращает найденные `ContextChunk` в текстовый context
- собирает prompt из инструкции, context и question
- отправляет prompt в OpenAI API
- печатает ответ модели
- печатает источники из использованного context, сгруппированные по файлам и chunk indexes

Запуск:

```bash
python -m src.ask "Когда была эндоскопия желудка?"
```

Пример формата источников:

```text
Источники:
- IMG_5371 2.txt: chunks 0, 1, 2
- IMG_5386.txt: chunks 0, 1
```

Источники формируются кодом из metadata найденных chunks, а не генерируются LLM. Это делает вывод стабильнее, но важно понимать ограничение: список показывает context, который был передан модели, а не точное доказательство того, какой именно chunk содержал конкретную фразу ответа.

Если retrieval не вернул context, CLI печатает `Контекст не найден` и не вызывает OpenAI API.

## Known Rough Edges

Что нужно будет улучшить после MVP:

- pure vector search может поставить OCR-мусор выше очевидного keyword match
- top-1 выдаче пока нельзя полностью доверять
- neighbor expansion повышает recall, но может подтягивать соседей от нерелевантных hits
- simple keyword boost помогает CLI-выдаче, но это не полноценный reranker
- короткие однословные запросы и запросы по именам/кличкам лучше обрабатывать hybrid search
- синонимы и аббревиатуры могут промахиваться из-за OCR-шума
- OCR-шум влияет и на embeddings, и на keyword matching
- итоговый context пока может содержать лишние источники; позже нужен лимит/threshold
- RAG-ответ зависит от качества OCR: модель может аккуратно восстанавливать очевидный смысл, но не должна выдумывать факты
- `src.ask` выводит источники использованного context, но пока не определяет точный chunk-доказательство для каждого факта ответа
- для offline-режима надёжнее использовать `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, а не `local_files_only=True` в коде

## Прогресс

**Готовность MVP: 90%**

```text
█████████░ 90%
```

- [x] OCR
- [x] Chunking
- [x] Embeddings
- [x] Retrieval MVP
- [x] Retrieval Eval
- [x] RAG Q&A

## Commit Stats

- Всего коммитов: 39
- Agent commits: 27
- User commits: 12

Agent commits считаются по префиксу `agent:` в commit message.

## Development Workflow

Проект развивается в гибридном формате:

- архитектурные и продуктовые решения принимает автор проекта
- часть реализации выполняется с помощью AI coding tools

Текущий commit split:

- AI-assisted commits: 27
- Manual commits: 12

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
- [x] Retrieval MVP
- semantic search over document chunks
- neighbor chunk expansion
- simple keyword boost for source ordering
- reusable `retrieve_context()` for CLI/eval
- [x] Retrieval Eval MVP
- local eval cases: query → expected source
- PASS/FAIL summary for retrieval regression checks
- [x] RAG Q&A MVP
- CLI question → retrieval context → prompt → OpenAI answer
- grouped source files/chunks in final CLI output
- no-context guard before LLM call


## Что Дальше

План развития проекта в сторону document AI / RAG.

- [ ] Add more sample documents
- build small local document archive

- [ ] Retrieval quality
- improve retrieval quality before relying on top-1
- add hybrid search: keyword/BM25 + vector search
- rerank top-k results by exact query term matches
- tune chunk expansion and context limits
- keep in mind: OCR noise and short name-based queries can make pure embeddings miss obvious chunks

- [ ] RAG Q&A polish
- distinguish likely answer chunks from broader context chunks
- add reusable `ask(question: str) -> str`
- add optional token usage/debug output

- [ ] OCR normalization experiments
- compare raw OCR vs LLM-cleaned OCR
- evaluate retrieval quality differences

- [ ] Optional future work
- SQLite archive/catalog
- metadata filtering
- structured extraction (JSON)
- local LLM support
