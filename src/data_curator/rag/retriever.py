from data_curator.embeddings.sentence_embedder import embed_query
from data_curator.vectorstore.qdrant_client import search


def _is_quality_chunk(text: str, min_alpha_ratio: float = 0.6, min_words: int = 50) -> bool:
    if not text or len(text.split()) < min_words:
        return False
    alpha_chars = sum(c.isalpha() or c.isspace() for c in text)
    return alpha_chars / len(text) >= min_alpha_ratio


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    query_vector = embed_query(question)
    results = search(query_vector, top_k=top_k * 3)
    quality_results = [r for r in results if _is_quality_chunk(r["chunk_text"])]
    return quality_results[:top_k]
# def retrieve(question: str, top_k: int =5) -> list[dict]:
#     query_vectory = embed_query(question)
#     return search(query_vectory, top_k=top_k)




