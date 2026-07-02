from src.data_curator.chunking.tokenizer import count_tokens, encode, decode
from src.data_curator.config import get_settings


SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

def _split_on_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    parts = text.split(separator)
    return [p+ separator for p in parts[:-1]] + [parts[-1]]

def _recursive_split(text: str, chunk_size:int,separators: list[str]) -> list[str]:
    if count_tokens(text) <= chunk_size:
        return [text]
    if not separators:
        return [text]
    sep, remaining_seps = separators[0], separators[1:]
    pieces = _split_on_separator(text, sep)

    result = []
    for piece in pieces:
        if count_tokens(piece) <= chunk_size:
            result.append(piece)
        else:
            result.extend(_recursive_split(piece, chunk_size, remaining_seps))
    return result


def _merge_with_overlap(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current_tokens: list[str] = []

    for piece in pieces:
        piece_tokens =encode(piece)
        if len(current_tokens) + len(piece_tokens) <= chunk_size:
            current_tokens.extend(piece_tokens)
        else:
            if current_tokens:
                chunks.append(decode(current_tokens))
            overlap_tokens =  current_tokens[-overlap:] if overlap > 0 else []
            current_tokens = overlap_tokens + piece_tokens
    if current_tokens:
        chunks.append(decode(current_tokens))
    return chunks


def recursive_character_split(
    text: str, chunk_size: int | None = None, overlap: int | None = None
) -> list[str]:
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    pieces = _recursive_split(text, chunk_size, SEPARATORS)
    return _merge_with_overlap(pieces, chunk_size, overlap)

