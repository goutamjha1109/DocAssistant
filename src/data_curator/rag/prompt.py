# src/data_curator/rag/prompt.py


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[{i+1}] (Paper: {chunk['title']}, score: {chunk['score']:.3f})\n"
        context += f"{chunk['chunk_text']}\n\n"

    system = (
        "You are a research assistant. Answer the user's question using only "
        "the provided context chunks. Cite which chunk number supports each "
        "part of your answer using [1], [2], etc. "
        "If the context does not contain enough information, say so explicitly."
    )

    user = f"Context:\n{context}\nQuestion: {question}"

    return system, user