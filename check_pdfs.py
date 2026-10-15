
import fitz
from pathlib import Path

for pdf in sorted(Path("data/raw").glob("*.pdf")):
    with fitz.open(pdf) as doc:
        text = "\n".join(page.get_text() for page in doc)

        print(f"\nFichier : {pdf.name}")
        print(f"Nombre de pages : {len(doc)}")
        print(f"Caractères extraits : {len(text)}")
        print(f"Début du texte :\n{text[:300]}")
