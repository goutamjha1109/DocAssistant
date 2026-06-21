import arxiv
import requests
from pathlib import Path

from transformers.testing_utils import parse_flag_from_env

from data_curator.config import get_settings
from pydantic import BaseModel, ConfigDict


# settings = get_settings()


class Paper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str
    published: str


def search_papers(query: str, max_results: int = 5) -> list[dict]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
        # sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers: list[Paper] = []
    for result in client.results(search):
        paper = Paper(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title,
            authors=[a.name for a in result.authors],
            abstract=result.summary,
            pdf_url=result.pdf_url,
            published=str(result.published.date())
        )

        papers.append(paper)
    return papers


def download_papers(paper: dict, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = download_dir / f"{paper.arxiv_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    response = requests.get(paper.pdf_url , stream=True)
    response.raise_for_status()
    with open(pdf_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
        return pdf_path


def save_metadata(paper: Paper, metadata_dir : Path) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{paper.arxiv_id}.json"
    metadata_path.write_text(paper.model_dump_json(indent=2), encoding="utf-8")
    return metadata_path

def load_metadata(arxiv_id: str, metadata_dir: Path) -> Path:
    metadata_path = metadata_dir / f"{arxiv_id}.json"
    return Paper.model_validate_json(metadata_path.read_text(encoding="utf-8"))




