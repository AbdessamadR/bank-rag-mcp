
import json
from pathlib import Path
import pymupdf

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

documents = []

for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
    with pymupdf.open(pdf_path) as pdf:
        year = next(
            (y for y in ("2023", "2024", "2025") if y in pdf_path.name),
            None
        )

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            documents.append({
                "source": pdf_path.name,
                "year": year,
                "page": page_number,
                "text": text
            })

            if not text:
                print(
                    f"Attention : page vide "
                    f"{page_number} dans {pdf_path.name}"
                )

output_path = OUTPUT_DIR / "pages.json"

with output_path.open("w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print("\nExtraction terminée.")
print(f"Nombre de pages : {len(documents)}")
print(f"Pages sans texte : {sum(not d['text'] for d in documents)}")
print(f"Caractères extraits : {sum(len(d['text']) for d in documents)}")
print(f"Fichier créé : {output_path}")
