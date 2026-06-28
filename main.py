import json
import os
from pathlib import Path
import argparse
from dotenv import load_dotenv

from src.data_curator.ingestion.arxiv_client import search_papers, download_papers, save_metadata
from src.data_curator.ingestion.pdf_parser import extract_text, save_parsed_text
from src.data_curator.ingestion.delete_pdf import delete_pdf
from src.data_curator.chunking.splitter import recursive_character_split
from src.data_curator.embeddings.sentence_embedder import embed_texts, embed_query
from src.data_curator.vectorstore.qdrant_client import ensure_collection, upsert_chunks, search
from src.data_curator.rag.retriever import retrieve
from src.data_curator.rag.prompt import build_prompt
from src.data_curator.rag.generator import generate
from src.data_curator.config import get_settings

load_dotenv(override=True)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
METADATA_DIR  = Path("data/metadata")
MANIFEST_PATH = Path("data/ingested.json")

CLOUD_RUN = os.getenv("CLOUD_RUN", "false").lower() == "true"

def load_manifest() -> set[str]:
    """Return set of already-ingested arxiv IDs."""
    if MANIFEST_PATH.exists():
        return set(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    return set()

def save_manifest(ingested: set[str]) -> None:
    """Persist the manifest — written after every successful paper."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(sorted(ingested), indent=2), encoding="utf-8"
    )


def ingest(query: str, max_results: int) -> None:
    settings = get_settings()
    ensure_collection(vector_size=settings.embedding_dim)

    ingested = load_manifest()
    papers = search_papers(query=query, max_results=max_results)

    new_count = 0
    for paper in papers:
        # Skip already-processed papers
        if paper.arxiv_id in ingested:
            print(f"already ingested: {paper.arxiv_id}:{paper.title}")
            continue
        try:
            pdf_path = download_papers(paper, download_dir=RAW_DIR)
            save_metadata(paper, METADATA_DIR)

            text = extract_text(pdf_path)
            save_parsed_text(text, paper.arxiv_id, PROCESSED_DIR)

            chunks = recursive_character_split(text)
            vectors = embed_texts(chunks)
            upsert_chunks(
                chunks=chunks,
                vectors=vectors,
                paper_id=paper.arxiv_id,
                title=paper.title,
                authors=paper.authors,
                source_pdf_path=str(pdf_path),
            )
            if CLOUD_RUN:
                deleted = delete_pdf(pdf_path=pdf_path)
                if deleted:
                    print(f"Successfully deleted pdf from local: {paper.title}:{paper.arxiv_id}")

            ingested.add(paper.arxiv_id)
            save_manifest(ingested)
            new_count += 1

        except Exception as e:
            print(f"ERROR processing {paper.arxiv_id}: {e}")
            continue

    if new_count == 0:
        print("No new papers to ingest.")
    else:
        print(f"\nDone. {new_count} new paper(s) ingested.")


def ask(question: str, top_k: int = 3) -> None:
    chunks = retrieve(question, top_k=top_k)
    system, user = build_prompt(question, chunks)
    answer = generate(system, user)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")
    print("\nSources:")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] {chunk['title']} (score={chunk['score']:.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", default="robotics", type=str)
    parser.add_argument("-d", "--max_results", default=2, type=int)
    parser.add_argument("--ask", type=str, default=None)
    args = parser.parse_args()

    if args.ask:
        ask(args.ask)
    else:
        ingest(query=args.query, max_results=args.max_results)


if __name__ == "__main__":
    main()


# def query_index(question: str, top_k: int = 3) -> None:
#     query_vec = embed_query(question)
#     results = search(query_vec, top_k=top_k)
#     for r in results:
#         print(f"score={r['score']:.3f} | {r['title']} | {r['chunk_text'][:150]}")
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-q", "--query", default="robotics", type=str)
#     parser.add_argument("-d", "--max_results", default=5, type=int)
#     parser.add_argument("--ask", type=str, default=None, help="Search the index instead of ingesting")
#     args = parser.parse_args()
#
#     if args.ask:
#         query_index(args.ask)
#     else:
#         ingest(query=args.query, max_results=args.max_results)

#
# if __name__ == "__main__":
#     main()