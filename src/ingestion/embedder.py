
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# 1. Définir les chemins du projet
ROOT_DIR = Path(__file__).resolve().parents[2]

CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks.json"
DB_PATH = ROOT_DIR / "data" / "vectorstore" / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "bank_reports"

BATCH_SIZE = 64


def main():
    # 2. Vérifier et charger les chunks
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CHUNKS_PATH}"
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    chunks = [
        chunk for chunk in chunks
        if chunk.get("text", "").strip()
    ]

    if not chunks:
        raise ValueError("Aucun chunk valide trouvé.")

    print(f"Chunks à indexer : {len(chunks)}")

    # 3. Charger le modèle d'embeddings
    print(f"Chargement du modèle : {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 4. Créer ou ouvrir la base vectorielle persistante
    DB_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 5. Générer et enregistrer les embeddings par lots
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        texts = [chunk["text"] for chunk in batch]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "source": str(chunk["source"]),
                    "year": str(chunk["year"]),
                    "page": int(chunk["page"]),
                    "chunk_index": int(chunk["chunk_index"])
                }
                for chunk in batch
            ]
        )

        print(
            f"Progression : {min(start + len(batch), total)}/{total}"
        )

    # 6. Vérifier le résultat
    print("\n=== INDEXATION TERMINÉE ===")
    print(f"Chunks traités : {total}")
    print(f"Vecteurs dans la collection : {collection.count()}")
    print(f"Base vectorielle : {DB_PATH}")


if __name__ == "__main__":
    main()
