# Génération des embeddings des chunks avec le modèle multilingue E5-Base (768 dimensions), puis indexation des vecteurs, des textes et de leurs métadonnées dans ChromaDB pour permettre la recherche sémantique.

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# 1. Configuration du projet
ROOT_DIR = Path(__file__).resolve().parents[2]

CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks.json"
DB_PATH = ROOT_DIR / "data" / "vectorstore" / "chroma_db"

MODEL_NAME = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "bank_reports_e5"

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

    # 3. Charger le modèle E5-Base
    print(f"Chargement du modèle : {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # 4. Connexion à ChromaDB
    DB_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # 5. Protection contre une collection existante
    existing_collections = [
        collection.name
        for collection in client.list_collections()
    ]

    if COLLECTION_NAME in existing_collections:
        raise ValueError(
            f"La collection {COLLECTION_NAME} existe déjà. "
            "Vérifie son contenu avant de relancer l'indexation."
        )

    # 6. Créer la nouvelle collection E5
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 7. Générer et enregistrer les embeddings par lots
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):

        batch = chunks[start:start + BATCH_SIZE]

        # Textes originaux
        texts = [
            chunk["text"]
            for chunk in batch
        ]

        # Préfixe obligatoire pour E5-Base
        texts_for_embedding = [
            "passage: " + text
            for text in texts
        ]

        # Génération des embeddings
        embeddings = model.encode(
            texts_for_embedding,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Enregistrement dans ChromaDB
        collection.upsert(
            ids=[
                chunk["chunk_id"]
                for chunk in batch
            ],
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
            f"Progression : "
            f"{min(start + len(batch), total)}/{total}"
        )

    # 8. Vérification finale
    indexed_count = collection.count()

    print("\n=== INDEXATION TERMINÉE ===")
    print(f"Modèle utilisé : {MODEL_NAME}")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Chunks source : {total}")
    print(f"Vecteurs indexés : {indexed_count}")
    print(f"Dimensions attendues : 768")
    print(f"Base vectorielle : {DB_PATH}")

    if indexed_count != total:
        raise RuntimeError(
            f"Nombre de vecteurs incorrect : "
            f"{indexed_count}/{total}"
        )


if __name__ == "__main__":
    main()
