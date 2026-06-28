# src/data_curator/rag/generator.py

from groq import Groq
from src.data_curator.config import get_settings

settings = get_settings()
_client = Groq(api_key=settings.groq_api_key)


def generate(system: str, user: str) -> str:
    response = _client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content