# src/data_curator/rag/prompt.py


# def build_prompt(question: str, chunks: list[dict]) -> str:
#     context = ""
#     for i, chunk in enumerate(chunks):
#         context += f"[{i+1}] (Paper: {chunk['title']}, score: {chunk['score']:.3f})\n"
#         context += f"{chunk['chunk_text']}\n\n"
#
#     system = (
#         "You are a research assistant. Answer the user's question using only "
#         "the provided context chunks. Cite which chunk number supports each "
#         "part of your answer using [1], [2], etc. "
#         "If the context does not contain enough information, say so explicitly."
#     )
#
#     user = f"Context:\n{context}\nQuestion: {question}"
#
#     return system, user

def build_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[{i+1}] (Paper: {chunk['title']}, score: {chunk['score']:.3f})\n"
        context += f"{chunk['chunk_text']}\n\n"

    system = (
        "You are a research assistant. Your job is to answer questions directly and concisely. "
        "Rules you must follow:\n"
        "1. Answer ONLY the question asked — do not ask follow-up questions.\n"
        "2. Use ONLY information from the provided context chunks.\n"
        "3. Cite chunk numbers inline using [1], [2], etc.\n"
        "4. If the context does not contain enough information, say 'The provided context does not contain enough information to answer this question.'\n"
        "5. Do not add greetings, sign-offs, or any text beyond the direct answer.\n"
        "6. Be concise — 3 to 5 sentences maximum."
    )

    user = f"Context:\n{context}\nQuestion: {question}\nAnswer:"

    return system, user