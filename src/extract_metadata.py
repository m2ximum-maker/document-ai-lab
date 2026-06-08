from __future__ import annotations

from pathlib import Path

from src.schemas import DocumentMetadata

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
CLEANED_OUTPUT_DIR = OUTPUT_DIR / "cleaned"
METADATA_OUTPUT_DIR = OUTPUT_DIR / "metadata"
METADATA_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "source",
        "owner",
        "doc_type",
        "doc_date",
        "doctor",
        "specialty",
        "clinic",
        "summary",
        "confidence",
        "warnings",
    ],
    "properties": {
        "source": {"type": "string"},
        "owner": {"type": ["string", "null"]},
        "doc_type": {
            "type": ["string", "null"],
            "enum": [
                "lab_test",
                "doctor_report",
                "prescription",
                "imaging",
                "procedure",
                "discharge_summary",
                "other",
                None,
            ],
        },
        "doc_date": {"type": ["string", "null"]},
        "doctor": {"type": ["string", "null"]},
        "specialty": {
            "type": ["string", "null"],
            "enum": [
                "gastroenterology",
                "otolaryngology",
                "neurology",
                "cardiology",
                "dermatology",
                "ophthalmology",
                "veterinary",
                "general_medicine",
                "other",
                None,
            ],
        },
        "clinic": {"type": ["string", "null"]},
        "summary": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}


def build_empty_metadata(source: str) -> DocumentMetadata:
    return {
        "source": source,
        "owner": None,
        "doc_type": None,
        "doc_date": None,
        "doctor": None,
        "specialty": None,
        "clinic": None,
        "summary": None,
        "confidence": 0.0,
        "warnings": [],
    }


def build_prompt(source: str, text: str) -> str:
    return f"""Извлеки metadata из OCR-текста медицинского документа.

Правила:
- Используй только информацию, которая явно присутствует в тексте.
- Не угадывай и не выдумывай данные.
- Если значение отсутствует или его нельзя определить достаточно уверенно, верни null.
- OCR-текст может содержать ошибки. Исправляй их только если правильный вариант очевиден из контекста.
- Не выполняй медицинскую интерпретацию, не ставь диагнозы и не делай выводы за врача.
- Извлекай факты из документа, а не предположения.

source:
Всегда верни "{source}".

owner:
Укажи имя пациента или субъекта документа так, как оно указано в тексте.
Для ветеринарных документов owner — животное, к которому относится медицинская информация, а не владелец животного.
Если субъект документа не указан или его нельзя уверенно определить, верни null.

doc_type:
Выбери только одно значение:
- "lab_test" — лабораторный анализ
- "doctor_report" — заключение врача, осмотр, консультация
- "prescription" — назначение лечения или рецепт
- "imaging" — рентген, УЗИ, МРТ, КТ и другие визуальные исследования
- "procedure" — процедура или инструментальное исследование, например ЭГДС, ФГДС, эндоскопия
- "discharge_summary" — выписка
- "other" — другой медицинский документ

Если тип нельзя определить, верни null.

specialty:
Выбери только одно значение:
- "gastroenterology"
- "otolaryngology"
- "neurology"
- "cardiology"
- "dermatology"
- "ophthalmology"
- "veterinary"
- "general_medicine"
- "other"

Если направление нельзя определить, верни null.

doc_date:
Дата документа в формате YYYY-MM-DD.
Если дата отсутствует или неоднозначна, верни null.

doctor:
ФИО врача или null.

clinic:
Название клиники, лаборатории или медицинской организации или null.

summary:
Краткое нейтральное описание документа на русском языке в 1–2 предложениях.
Не интерпретируй результаты и не делай медицинских выводов.

confidence:
Число от 0 до 1, общая уверенность в извлечённых metadata.

warnings:
Список строк с OCR-проблемами, сомнениями или неоднозначностями.
Если проблем нет, верни пустой список.

OCR-текст:
{text}
"""


def find_cleaned_documents() -> list[Path]:
    if not CLEANED_OUTPUT_DIR.exists():
        return []

    return sorted(CLEANED_OUTPUT_DIR.glob("*.txt"))


def main() -> None:
    METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = find_cleaned_documents()

    print(f"Найдено документов: {len(documents)}")
    print(f"Metadata output: {METADATA_OUTPUT_DIR.relative_to(ROOT_DIR)}")

    for path in documents:
        metadata = build_empty_metadata(path.name)
        print(f"- {metadata['source']}")


if __name__ == "__main__":
    main()
