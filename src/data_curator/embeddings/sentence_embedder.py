from sentence_transformers import SentenceTransformer
from data_curator.config import get_settings

settings = get_settings()
_model = SentenceTransformer(settings.sentence_transformer_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = _model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]