import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
model = os.getenv("OPENAI_MODEL")

if model is None:
    raise ValueError("OPENAI_MODEL не установлена")


response = client.responses.create(model=model, input="Привет! Какие новости?")

print(response.output_text)

if response.usage:
    print("input tokens:", response.usage.input_tokens)
    print("output tokens:", response.usage.output_tokens)
    print("total tokens:", response.usage.total_tokens)
