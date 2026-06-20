import arxiv
import requests
from pathlib import Path
# from data_curator.config import get_settings
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


def search_papers(query: str, max_results: int = 5) -> list[Paper]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
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


def download_papers(paper: Paper, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = download_dir / f"{paper.arxiv_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    response = requests.get(paper.pdf_url , stream=True)
    try:
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
            return pdf_path
    except Exception as e:
        print(f"Error occured while downloading paper : {paper.title}:{e}")


