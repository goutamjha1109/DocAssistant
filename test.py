from pathlib import Path
import argparse

from data_curator.vectorstore.qdrant_client import ensure_collection, upsert_chunks, search
from data_curator.embeddings.sentence_embedder import embed_texts, embed_query
from data_curator.config import get_settings
from pathlib import Path
from data_curator.chunking.splitter import recursive_character_split
from data_curator.embeddings.sentence_embedder import embed_texts


from data_curator.ingestion.arxiv_client import (
    search_papers,
    download_papers, Paper
)

from data_curator.ingestion.pdf_parser import (
    extract_text,
    save_parsed_text
)

def fetch_arxiv_id(query: str, max_results : int) -> Path:
    papers = search_papers(
        query=query,
        max_results=max_results
    )

    for paper in papers:
        pdf_path = download_papers(
            paper,
            download_dir=Path("data/raw")
        )
        pages_text = extract_text(pdf_path)
        output_path = save_parsed_text(pages_text, paper.arxiv_id, Path("data/processed"))
        print(f"Papers parse and saved to {output_path}")
    print(f"Downloaded: {paper.title}")
    print(f"Raw Papers Saved to: {pdf_path}")

    print("All papers downloaded successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", default="robotics", type=str, help="query to search")
    parser.add_argument("-d", "--max_results", default=5, type=int, help="max number of papers to download")
    args = parser.parse_args()
    # fetch_arxiv_id(query=args.query, max_results=args.max_results)


    settings = get_settings()

    file_path = Path().cwd() / "data" / "processed" / "2606.20549v1.txt"
    text = file_path.read_text(encoding="utf-8")
    chunks = recursive_character_split(text)

    ensure_collection(vector_size=settings.embedding_dim)

    vectors = embed_texts(chunks)  # all chunks from before, not just 3
    upsert_chunks(
        chunks=chunks,
        vectors=vectors,
        paper_id="2606.20549v1",
        title="Generating Robot Hands from Human Demonstrations",
        authors=["..."],
        source_pdf_path="data/raw/2606.20549v1.pdf",
    )

    query_vec = embed_query("How does the robot hand design framework optimize hardware?")
    results = search(query_vec, top_k=3)
    for r in results:
        print(f"score={r['score']:.3f} | {r['chunk_text'][:150]}")


if __name__ == "__main__":
    main()

