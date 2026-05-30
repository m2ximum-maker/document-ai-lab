import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from src.search import retrieve_context, ContextChunk

load_dotenv()

client = OpenAI()
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

    Контекст:
    {context_text}

    Вопрос:
    {question}
    """


def ask_llm(prompt: str) -> str:
    if model is None:
        raise ValueError("OPENAI_MODEL не установлена")

    response = client.responses.create(model=model, input=prompt)

    if response.usage:
        print("input tokens:", response.usage.input_tokens)
        print("output tokens:", response.usage.output_tokens)
        print("total tokens:", response.usage.total_tokens)

    return response.output_text


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

    if not search_result.context:
        print("Контекст не найден")
        return

    prompt = build_prompt(question, context_text)
    answer = ask_llm(prompt)
    print(answer)

    if not search_result.context:
        print("Контекст не найден")
        return


if __name__ == "__main__":
    main()
