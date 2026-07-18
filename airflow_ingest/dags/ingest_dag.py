from airflow.sdk import dag, task, Context,chain
from airflow.sdk.bases.sensor import PokeReturnValue
from datetime import datetime
import json
from pathlib import Path
import os

@dag(
    start_date = datetime(year=2026,month=7,day=14),
    description = "DAG to run Ingestion and Embedding creation using airflow.",
    schedule = "@weekly",
    max_consecutive_failed_dag_runs = 3,
    tags=['rag']
)
def ingest_arxiv_papers():
    
    @task.sensor(poke_interval=30, timeout=3600, mode="poke")
    def ensure_qdrant_collection() -> PokeReturnValue :
        from src.data_curator.config import get_settings
        from src.data_curator.vectorstore.qdrant_client import ensure_collection
        
        settings = get_settings()
        try:
            ensure_collection(vector_size=settings.embedding_dim)
            return PokeReturnValue(is_done=True)
        
        except Exception as e:
            print(f"[ensure_qdrant_collection] Failed setting up qdrant colelction:{e}")
            return PokeReturnValue(is_done=False)

    
    @task
    def load_metadata_json(**context:Context) -> None:
        """Returns list of already-ingested paper metadata dicts."""
        METADATA_PATH = Path(r"data\metadata\ingested_papers.json")
        if METADATA_PATH.exists():
            ingested_papers = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        else:
            ingested_papers = []
        context['ti'].xcom_push(key='ingested_papers',value=ingested_papers)
    
    @task
    def get_new_papers_from_arxiv(**context:Context) -> None:
        from src.data_curator.ingestion.arxiv_client import search_papers,Paper
        
        papers:list[Paper] = search_papers(query='robotics')
        already_ingested_papers:list[dict] = context['ti'].xcom_pull(task_ids='load_metadata_json',key='ingested_papers')
        ingested_paper_ids = [ingested_paper['arxiv_id'] for ingested_paper in already_ingested_papers]
        new_papers: list[dict] = [paper.model_dump() for paper in papers if paper.arxiv_id not in ingested_paper_ids]
        return new_papers
    
    @task
    def process_single_paper(paper,**context:Context):
        from src.data_curator.ingestion.arxiv_client import download_papers
        from src.data_curator.ingestion.pdf_parser import extract_text
        from src.data_curator.chunking.splitter import recursive_character_split
        from src.data_curator.vectorstore.qdrant_client import upsert_chunks
        from src.data_curator.embeddings.sentence_embedder import embed_texts
        from src.data_curator.ingestion.delete_pdf import delete_pdf
        from src.data_curator.ingestion.arxiv_client import save_metadata
        from src.data_curator.ingestion.arxiv_client import Paper
        
        RAW_DIR = Path("data/raw")
        METADATA_PATH = Path(r"data\metadata\ingested_papers.json")
        CLOUD_RUN = os.getenv("CLOUD_RUN", "false").lower() == "true"
        
        already_ingested_papers:list[dict] = context['ti'].xcom_pull(task_ids='load_metadata_json',key='ingested_papers')
        
        paper = Paper(**paper)
        pdf_path = download_papers(paper, download_dir=RAW_DIR)
        text:str = extract_text(pdf_path)
        chunks:list[str] = recursive_character_split(text)
        vectors:list[list[float]] = embed_texts(chunks)
        upsert_chunks(
                chunks=chunks,
                vectors=vectors,
                paper_id=paper.arxiv_id,
                title=paper.title,
                authors=paper.authors,
                source_pdf_path=str(pdf_path),
                )
                
        if CLOUD_RUN:
            delete_pdf(pdf_path=pdf_path)
            
        already_ingested_papers.append(paper.model_dump())
        
        save_metadata(already_ingested_papers, METADATA_PATH)
        
    
    ensure_qdrant_collection() >> load_metadata_json() >> get_new_papers_from_arxiv() >> process_single_paper.partial().expand(paper=get_new_papers_from_arxiv())

ingest_arxiv_papers()