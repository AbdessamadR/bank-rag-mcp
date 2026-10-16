
import json
from pathlib import Path
from typing import Any

INPUT_PATH = Path("data/processed/pages.json")
OUTPUT_PATH = Path("data/processed/chunks.json")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Découpe un texte en passages avec chevauchement."""

    if chunk_size <= 0:
        raise ValueError("chunk_size doit être positif")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap doit être compris entre 0 et chunk_size - 1")

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def create_chunks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Crée des chunks en conservant les métadonnées des pages."""

    all_chunks = []

    for document in pages:
        for index, chunk_text in enumerate(split_text(document["text"]), start=1):
            all_chunks.append({
                "chunk_id": (
                    f'{document["year"]}_'
                    f'{document["page"]}_{index}'
                ),
                "source": document["source"],
                "year": document["year"],
                "page": document["page"],
                "chunk_index": index,
                "text": chunk_text,
            })

    return all_chunks


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        pages = json.load(file)

    chunks = create_chunks(pages)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    total_chars = sum(len(chunk["text"]) for chunk in chunks)
    average_size = total_chars / len(chunks) if chunks else 0

    print("=== CHUNKING TERMINÉ ===")
    print(f"Pages traitées : {len(pages)}")
    print(f"Chunks créés : {len(chunks)}")
    print(f"Taille moyenne : {average_size:.1f} caractères")
    print(f"Fichier créé : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
