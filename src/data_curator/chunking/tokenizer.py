import tiktoken
import os

from src.data_curator.config import get_settings
settings =  get_settings()



_encoder = tiktoken.encoding_for_model(settings.embedding_model)


def count_tokens(text: str) -> int:
    return len(encode(text))

def encode(text: str) -> list[int]:
    return _encoder.encode(text)

def decode(tokens: list[int]) -> str:
    return _encoder.decode(tokens)

def decode_single_tokens(tokens:list[int]) -> list[str]:
    return [_encoder.decode_single_token_bytes(token) for token in tokens]