import tiktoken
import os

from data_curator.config import get_settings
settings =  get_settings()



_encoder = tiktoken.encoding_for_model(settings.embedding_model)


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))

def encode(text: str) -> list[int]:
    return _encoder.encode(text)

def decode(tokens: list[int]) -> str:
    return _encoder.decode(tokens)


