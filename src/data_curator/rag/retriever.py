from data_curator.embeddings.sentence_embedder import embed_query
from data_curator.vectorstore.qdrant_client import search


def retrieve(question: str, top_k: int =3) -> list[dict]:
    query_vectory = embed_query(question)
    return search(query_vectory, top_k=top_k)




