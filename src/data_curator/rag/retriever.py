from src.data_curator.embeddings.sentence_embedder import embed_query
from src.data_curator.vectorstore.qdrant_client import search, fetch_all_points_from_qdrant
from src.data_curator.rag.bm_score import BM25Scorer
from src.data_curator.rag.reranker import rerank_retrieved_chunks

def dense_retriever(question: str, top_k: int = 20) -> list[dict]:
    query_vector = embed_query(question)
    return search(query_vector, top_k=top_k)

def sparse_retriever(question: str,top_k: int = 20) -> list[dict]:
    all_chunks = fetch_all_points_from_qdrant()
    scorer = BM25Scorer(query=question,chunks = all_chunks)
    scored_chunks = scorer.get_bm25_scores(top_k=top_k)
    return scored_chunks

def reciprocal_rank_fusion(
    dense_retrieved_chunks: list[dict], sparse_retrieved_chunks: list[dict],top_k:int = 10, k: int = 60
):
    ranked_dense_retrieved_chunks = []
    for rank, chunk in enumerate(dense_retrieved_chunks,1):
        fusion_index = f"{chunk['paper_id']}_{chunk['chunk_index']}"
        chunk.update({'rank':rank,'fusion_index':fusion_index})
        ranked_dense_retrieved_chunks.append(chunk)

    ranked_sparse_retrieved_chunks = []
    for rank, chunk in enumerate(sparse_retrieved_chunks,1):
        fusion_index = f"{chunk['paper_id']}_{chunk['chunk_index']}"
        chunk.update({'rank':rank,'fusion_index':fusion_index})
        ranked_sparse_retrieved_chunks.append(chunk)

    unique_ranked_dense_indices = set(chunk['fusion_index'] for chunk in ranked_dense_retrieved_chunks)
    unique_ranked_sparse_indices = set(chunk['fusion_index'] for chunk in ranked_sparse_retrieved_chunks)

    all_unique_chunk_indices = unique_ranked_dense_indices | unique_ranked_sparse_indices

    rrf_fused = []
    for index in all_unique_chunk_indices:
        rrf_score = 0

        if index in unique_ranked_dense_indices:
            chunk_item = [chunk for chunk in ranked_dense_retrieved_chunks 
                    if chunk['fusion_index'] == index][0]
            rank = chunk_item['rank']
            rrf_score += 1/(k+rank)

        if index in unique_ranked_sparse_indices:
            chunk_item = [chunk for chunk in ranked_sparse_retrieved_chunks 
                    if chunk['fusion_index'] == index][0]
            rank = chunk_item['rank']
            rrf_score += 1/(k+rank)

        rrf_fused.append(
            {
                "chunk_text": chunk_item["chunk_text"],
                "chunk_index": chunk_item["chunk_index"],
                "rrf_score": rrf_score,
                "paper_id": chunk_item["paper_id"],
                "title": chunk_item["title"],
            }
        )

    return sorted(rrf_fused, key = lambda x: x['rrf_score'], reverse = True)[:top_k]


def hybrid_retriever(
    question: str, 
    dense_top_k: int = 20, 
    sparse_top_k: int = 20, 
    rrf_top_k: int = 10,
    final_top_k: int = 5
) -> list[dict]:
    dense_retrieved_chunks = dense_retriever(question, top_k=dense_top_k)
    sparse_retrieved_chunks = sparse_retriever(question, top_k=sparse_top_k)

    rrf_combined_chunks = reciprocal_rank_fusion(
        dense_retrieved_chunks = dense_retrieved_chunks,
        sparse_retrieved_chunks = sparse_retrieved_chunks,
        top_k = rrf_top_k
    )

    chunks = [chunk['chunk_text'] for chunk in rrf_combined_chunks]
    reranked_chunks = rerank_retrieved_chunks(
        query=question, chunks=chunks, top_k=final_top_k
    )

    final_output = []
    for item in reranked_chunks:
        original_chunk = rrf_combined_chunks[item['corpus_id']]
        final_output.append(
            {
                "title": original_chunk.get("title"),
                "paper_id": original_chunk.get("paper_id"),
                "chunk_index": original_chunk.get("chunk_index"),
                "chunk_text": item["text"],
                "score": item["score"],
            }
        )
    return final_output
