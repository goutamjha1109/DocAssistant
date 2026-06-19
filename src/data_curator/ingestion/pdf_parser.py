import fitz
from pathlib import Path




def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()
    return "\n".join(pages_text)


def save_parsed_text(text: str, arxiv_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{arxiv_id}.txt"
    output_path.write_text(text, encoding="utf-8")
    return output_path


