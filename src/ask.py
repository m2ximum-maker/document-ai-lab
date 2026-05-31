import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from src.search import retrieve_context, ContextChunk

load_dotenv()

model = os.getenv("OPENAI_MODEL")


def build_context_text(chunks: list[ContextChunk]) -> str:
    parts = []

    for chunk in chunks:
        part = f"""source: {chunk.source}
chunk_index: {chunk.chunk_index}
text:
{chunk.document}"""
        parts.append(part)

    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, context_text: str) -> str:
    return f"""Ответь на вопрос пользователя, используя только контекст ниже.

Если в контексте нет ответа, скажи: "В найденном контексте нет ответа."

Важно:
- отвечай нормальным русским языком;
- не копируй OCR-ошибки и случайные символы;
- если текст OCR повреждён, восстанови смысл только там, где он очевиден;
- не выдумывай факты, которых нет в контексте.

Контекст:
{context_text}

Вопрос:
{question}
"""


def get_client() -> OpenAI:
    return OpenAI()


def ask_llm(prompt: str) -> str:
    if model is None:
        raise ValueError("OPENAI_MODEL не установлена")

    client = get_client()
    response = client.responses.create(model=model, input=prompt)

    return response.output_text


def format_sources(chunks: list[ContextChunk]) -> str:
    sources: dict[str, set[int]] = {}

    for chunk in chunks:
        if chunk.source not in sources:
            sources[chunk.source] = set()

        sources[chunk.source].add(chunk.chunk_index)

    lines = ["Источники:"]

    for source, chunk_indexes in sources.items():
        indexes_text = ", ".join(str(index) for index in sorted(chunk_indexes))
        lines.append(f"- {source}: chunks {indexes_text}")

    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print('Использование: python -m src.ask "ваш вопрос"')
        return

    question = " ".join(sys.argv[1:])
    search_result = retrieve_context(question)

    if not search_result.context:
        print("Контекст не найден")
        return

    context_text = build_context_text(search_result.context)
    prompt = build_prompt(question, context_text)
    # answer = ask_llm(prompt)
    answer = "Гастроскопию Степанову Максиму Руслановичу 26.12.2023 делал врач Углов Роман Михайлович."
    print(answer)
    print(format_sources(search_result.context))


if __name__ == "__main__":
    main()
