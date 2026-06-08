# Прогресс

Этот файл хранит историю состояния проекта и ближайшие шаги. README остаётся короткой картой проекта, а подробное направление описано в [PROJECT_DIRECTION.md](PROJECT_DIRECTION.md).

## Текущее Состояние

MVP OCR/RAG уже работает:

- [x] OCR изображений
- [x] сохранение raw/cleaned OCR-текста
- [x] chunking из `output/cleaned/*.txt`
- [x] сохранение `output/chunks/chunks.jsonl`
- [x] embeddings в Chroma
- [x] hybrid retrieval: vector search + BM25 + RRF
- [x] retrieval eval baseline
- [x] RAG Q&A через `src.ask`
- [x] базовый SQLite Catalog через `src.catalog`
- [x] scaffold для metadata extraction через `src.extract_metadata`
- [ ] LLM metadata extraction в `output/metadata/*.json`
- [ ] фильтрация по владельцу, дате, типу документа, врачу, клинике

## Текущий Pipeline

```text
input images -> OCR -> output/cleaned/*.txt -> chunk.py -> chunks.jsonl -> Chroma -> search/ask
```

Параллельные структурные ветки:

```text
output/cleaned/*.txt
├── chunk.py -> chunks.jsonl -> Chroma -> search/ask
├── catalog.py -> output/catalog/catalog.db
└── extract_metadata.py -> output/metadata/*.json (следующий шаг)
```

## Следующий Большой Блок

Добавить LLM metadata extraction рядом с текущим flow.

Цель metadata extraction:

- читать `output/cleaned/*.txt`
- отправлять OCR-текст в LLM
- получать structured JSON по `DocumentMetadata`
- сохранять отдельные файлы `output/metadata/*.json`
- не писать в SQLite на этом шаге
- не ломать существующий Chroma-based поиск и RAG

Существующий RAG сохраняется. `chunk.py`, `search.py`, `ask.py`, `chunks.jsonl` и Chroma не меняются, пока metadata extraction не будет отдельно проверена.

## Уже Сделано По Каталогу

`src.catalog` уже:

- создаёт `output/catalog/catalog.db`
- создаёт таблицу `documents`
- читает `output/cleaned/*.txt`
- добавляет документы по `source`
- запускается через `python -m src.catalog`
- не подключён к `search.py` и `ask.py`

## План Маленькими Шагами

### Шаг 0: приватность владельцев

- [x] хранить реальные имена только в локальном `src/private_owners.json`
- [x] держать `src/private_owners.json` вне git и не добавлять его в историю
- [x] коммитить только обезличенный пример `src/private_owners.example.json`
- [ ] в extracted metadata сохранять псевдонимы вроде `person_1`, `cat_1`, а не реальные имена
- [ ] перед пушем проверять, что приватный словарь и metadata JSON не попали в tracked files

### Шаг 1: SQLite Catalog MVP

- [x] посмотреть текущую структуру проекта
- [x] предложить минимальную SQLite-схему
- [x] обсудить `documents`
- [x] не добавлять таблицу `chunks` на первом шаге
- [x] добавить `src/catalog.py`
- [x] создавать SQLite базу `output/catalog/catalog.db`
- [x] создавать таблицу `documents`
- [x] читать `output/cleaned/*.txt`
- [x] делать insert с `ON CONFLICT(source) DO NOTHING`
- [x] добавить запуск `python -m src.catalog`
- [x] не подключать это к `search.py` и `ask.py`

### Шаг 2: Metadata Extraction Scaffold

- [x] добавить `src/extract_metadata.py`
- [x] добавить `output/metadata/.gitkeep`
- [x] игнорировать `output/metadata/*`
- [x] добавить `DocumentMetadata`
- [x] добавить `METADATA_JSON_SCHEMA`
- [x] сделать `confidence` числом от `0` до `1`
- [x] добавить `build_prompt(source, text)`
- [x] запускать `python -m src.extract_metadata`
- [x] пока только печатать найденные cleaned-документы

### Шаг 3: Первый LLM Прогон

- [ ] добавить OpenAI client в `src.extract_metadata`
- [ ] использовать `.env`: `OPENAI_API_KEY`, `OPENAI_MODEL`
- [ ] отправить один cleaned-документ в LLM
- [ ] использовать structured output через `METADATA_JSON_SCHEMA`
- [ ] распарсить ответ через `json.loads`
- [ ] сохранить `output/metadata/<source-stem>.json`
- [ ] убедиться, что metadata JSON ignored и не попадает в git
- [ ] не менять `chunk.py`, `search.py`, `ask.py`

### Шаг 4: Удобный CLI Для Metadata

- [ ] добавить `--limit`
- [ ] добавить `--force`
- [ ] пропускать существующие JSON без `--force`
- [ ] печатать created/skipped/errors
- [ ] обрабатывать ошибки одного документа без остановки всего запуска

### Шаг 5: Owner Normalization

- [ ] загрузить локальный `src/private_owners.json`, если он есть
- [ ] заменить реальные имена в `owner` на псевдонимы
- [ ] не сохранять реальные owner names в metadata JSON
- [ ] добавить warning, если owner найден в тексте, но не найден в словаре

## Guardrails

Пока не делаем:

- не переписываем архитектуру целиком
- не переписываем текущий RAG
- не пишем LLM metadata напрямую в SQLite
- не мигрируем Chroma
- не меняем формат `chunks.jsonl`
- не меняем существующий поиск без отдельного обсуждения
- не меняем `ask.py` без отдельного обсуждения

## Идеи На Потом

- таблица или слой для extracted metadata
- confidence для каждого извлечённого поля
- нормализация имён врачей и клиник
- фильтры по владельцу, году, типу документа, врачу
- связь `documents` с chunks
- отображение источников на уровне документа, страницы и chunk
- поддержка PDF
- UI поверх поиска и фильтров
