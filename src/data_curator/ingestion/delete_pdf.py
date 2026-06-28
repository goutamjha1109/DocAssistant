from pathlib import Path

def delete_pdf(pdf_path: str) -> bool:
    try:
        Path(pdf_path).unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"Error deleting PDF {pdf_path}: {e}")
        return False
