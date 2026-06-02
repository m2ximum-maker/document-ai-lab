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
- [ ] SQLite Catalog
- [ ] metadata extraction
- [ ] фильтрация по владельцу, дате, типу документа, врачу, клинике

## Текущий Pipeline

```text
input images -> OCR -> output/cleaned/*.txt -> chunk.py -> chunks.jsonl -> Chroma -> search/ask
```

## Следующий Большой Блок

Добавить SQLite Catalog рядом с текущим flow.

Цель каталога:

- хранить документы как объекты
- связать файл с владельцем, датой, типом документа и будущими metadata
- подготовить основу для фильтрации и более надёжных ответов
- не ломать существующий Chroma-based поиск

SQLite добавляется как соседняя ветка:

```text
output/cleaned/*.txt
├── chunk.py -> chunks.jsonl -> Chroma -> search/ask
└── catalog.py -> SQLite
```

Существующий RAG сохраняется. `search.py` и `ask.py` не меняются, пока каталог не будет создан, заполнен и отдельно обсуждён.

## План Маленькими Шагами

### Шаг 1: схема

- [ ] посмотреть текущую структуру проекта
- [ ] предложить минимальную SQLite-схему
- [ ] обсудить `documents`
- [ ] обсудить, нужна ли таблица `chunks` сейчас или позже
- [ ] не менять код без подтверждения

### Шаг 2: `src/db.py`

- [ ] добавить минимальный модуль `src/db.py`
- [ ] создавать SQLite базу `output/catalog/catalog.db`
- [ ] создавать таблицу `documents`
- [ ] добавить функцию `init_db()`
- [ ] не подключать это к `search.py` и `ask.py`

### Шаг 3: заполнение каталога

- [ ] добавить `src/catalog.py` или `src/ingest_catalog.py`
- [ ] читать `output/cleaned/*.txt`
- [ ] создавать или обновлять записи `documents`
- [ ] сделать upsert по `source` или `filename`
- [ ] не менять `chunk.py`, `search.py`, `ask.py`

Начальные поля:

- `id`
- `source`
- `filename`
- `owner`
- `doc_type`
- `doc_date`
- `created_at`
- `updated_at`

### Шаг 4: CLI

- [ ] добавить запуск `python -m src.catalog`
- [ ] инициализировать базу
- [ ] заполнить `documents`
- [ ] вывести количество добавленных и обновлённых документов

## Guardrails

Пока не делаем:

- не переписываем архитектуру целиком
- не переписываем текущий RAG
- не внедряем LLM metadata extraction
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
