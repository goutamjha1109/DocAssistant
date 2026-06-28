
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.data_curator.config import get_settings

settings = get_settings()
_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def ensure_collection(vector_size: int) -> None:
    if not _client.collection_exists(settings.qdrant_collection):
        _client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _make_point_id(paper_id: str, chunk_index: int) -> str:
    # deterministic id -> re-ingesting the same paper overwrites instead of duplicating
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{paper_id}_{chunk_index}"))


def upsert_chunks(
    chunks: list[str],
    vectors: list[list[float]],
    paper_id: str,
    title: str,
    authors: list[str],
    source_pdf_path: str,
) -> None:
    points = [
        PointStruct(
            id=_make_point_id(paper_id, idx),
            vector=vector,
            payload={
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "chunk_index": idx,
                "source_pdf_path": source_pdf_path,
                "chunk_text": chunk_text,
            },
        )
        for idx, (chunk_text, vector) in enumerate(zip(chunks, vectors))
    ]
    _client.upsert(collection_name=settings.qdrant_collection, points=points)


def search(query_vector: list[float], top_k: int = 5) -> list[dict]:
    results = _client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        {
            "score": r.score,
            "paper_id": r.payload["paper_id"],
            "title": r.payload["title"],
            "chunk_index": r.payload["chunk_index"],
            "chunk_text": r.payload["chunk_text"],
        }
        for r in results
    ]