from airflow.sdk import dag, task, Context
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
    """DAG to ingest papers from Arxiv. Use arxiv client to retrieve papers from arxiv.
    Download new papers, split them into chunks, create embeddings and store in qdrant db.

    Returns:
        None
    """
    
    @task.sensor(poke_interval=30, timeout=3600, mode="poke")
    def ensure_qdrant_collection() -> PokeReturnValue :
        """Sensor to ensure if qdrant collection is created

        Returns:
            PokeReturnValue: Returns true post collection creation
        """
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
        """Pushes already-ingested paper metadata dicts into xcom"""
        METADATA_PATH = Path("data/metadata/ingested_papers.json")
        if METADATA_PATH.exists():
            ingested_papers = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        else:
            ingested_papers = []
        context['ti'].xcom_push(key='ingested_papers',value=ingested_papers)
    
    @task
    def get_new_papers_from_arxiv(**context:Context) -> list[dict]:
        """Finds new papers to process and store in qdrantdb

        Returns:
            list[dict]: list of paper objects
        """
        from src.data_curator.ingestion.arxiv_client import search_papers,Paper
        
        already_ingested_papers:list[dict] = context['ti'].xcom_pull(task_ids='load_metadata_json',key='ingested_papers')
        ingested_paper_ids = [ingested_paper['arxiv_id'] for ingested_paper in already_ingested_papers]
        papers:list[Paper] = search_papers(query='robotics')
        new_papers: list[dict] = [paper.model_dump() for paper in papers if paper.arxiv_id not in ingested_paper_ids]
        return new_papers
    
    @task
    def process_single_paper(paper:dict) -> dict:
        """Downloads pdf paper locally, parses pdf, create chunks, vectorize them and stores in db.

        Args:
            paper (dict): paper object

        Returns:
            dict: same paper object if successful
        """
        from src.data_curator.ingestion.arxiv_client import download_papers
        from src.data_curator.ingestion.pdf_parser import extract_text
        from src.data_curator.chunking.splitter import recursive_character_split
        from src.data_curator.vectorstore.qdrant_client import upsert_chunks
        from src.data_curator.embeddings.sentence_embedder import embed_texts
        from src.data_curator.ingestion.delete_pdf import delete_pdf
        from src.data_curator.ingestion.arxiv_client import Paper
        
        RAW_DIR = Path("data/raw")
        CLOUD_RUN = os.getenv("CLOUD_RUN", "false").lower() == "true"
        
        paper_obj = Paper(**paper)
        pdf_path = download_papers(paper_obj, download_dir=RAW_DIR)
        text:str = extract_text(pdf_path)
        chunks:list[str] = recursive_character_split(text)
        vectors:list[list[float]] = embed_texts(chunks)
        upsert_chunks(
                chunks=chunks,
                vectors=vectors,
                paper_id=paper_obj.arxiv_id,
                title=paper_obj.title,
                authors=paper_obj.authors,
                source_pdf_path=str(pdf_path),
                )
                
        if CLOUD_RUN:
            delete_pdf(pdf_path=pdf_path)
        
        return paper
    
    @task(trigger_rule='all_done')
    def update_metadata(processed_papers:list[dict],**context:Context):
        """Updates metadata json and adds new processed pdfs (arxiv id, paper title, etc.)
        Args:
            processed_papers (list[dict]): processed_papers
        """
        from src.data_curator.ingestion.arxiv_client import save_metadata
        METADATA_PATH = Path("data/metadata/ingested_papers.json")
        processed_papers = [paper for paper in processed_papers if paper is not None]
        
        ingested_papers = context['ti'].xcom_pull(task_ids='load_metadata_json',key='ingested_papers')
        ingested_papers.extend(processed_papers)
        save_metadata(ingested_papers, METADATA_PATH)

    
    collection_ensured = ensure_qdrant_collection()
    metadata = load_metadata_json()
    new_papers = get_new_papers_from_arxiv()
    processed_papers  = process_single_paper.partial().expand(paper=new_papers)
    updated_metadata = update_metadata(processed_papers)
    
    collection_ensured >> metadata >> new_papers >> processed_papers >> updated_metadata

ingest_arxiv_papers()