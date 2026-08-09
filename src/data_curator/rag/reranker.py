from sentence_transformers import CrossEncoder
from src.data_curator.config import get_settings

settings = get_settings()
_cross_encoder_model = CrossEncoder(f"cross-encoder/{settings.cross_encoder_model}")


def rerank_retrieved_chunks(
    query: str, chunks: list[str], top_k: int = 10
) -> list[dict]:
    return _cross_encoder_model.rank(
        query=query, documents=chunks, return_documents=True, top_k=top_k
    )


# if __name__ == "__main__":
#     from src.data_curator.chunking.splitter import recursive_character_split
    
#     path = r"D:\DocAssistant\data\processed\2606.20527v1.txt"
#     with open(path,'r',encoding='utf-8') as file:
#         text = file.read()
    
#     chunks = recursive_character_split(text)
#     query = "What is the definition of StylisticBias as introduced in the paper?"
#     items = rerank_retrieved_chunks(query,chunks)
    
#     for item in items:
#         print("="*10)
#         print(item['text'])
#         print(item['score'])
#         print("="*10)