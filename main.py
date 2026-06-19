from pathlib import Path
import argparse

import pandas as pd

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
    fetch_arxiv_id(query=args.query, max_results=args.max_results)


if __name__ == "__main__":
    main()

