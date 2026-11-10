
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_PATH = (
    PROJECT_ROOT / "data" / "vectorstore" / "chroma_db"
)

# Configuration E5-Base
MODEL_NAME = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "bank_reports_e5"
QUERY_PREFIX = "query: "

EXPECTED_CHUNKS = 3992

REQUIRED_METADATA = [
    "source",
    "year",
    "page",
    "chunk_index",
]

BATCH_SIZE = 500

TEST_QUERY = (
    "Quels sont les résultats financiers "
    "et le chiffre d'affaires de la banque ?"
)


# ============================================================
# 2. AUDIT DE LA COLLECTION
# ============================================================

def audit_collection(collection, model):

    total = collection.count()

    print("\n" + "-" * 60)
    print(f"Collection : {collection.name}")
    print(f"Nombre de vecteurs : {total}")

    if total == EXPECTED_CHUNKS:
        print("OK : nombre conforme aux 3992 chunks.")
    else:
        print(
            f"ATTENTION : {total} vecteurs "
            f"(attendu : {EXPECTED_CHUNKS})."
        )

    # --------------------------------------------------------
    # 3. CONTRÔLE DES DOCUMENTS ET DES MÉTADONNÉES
    # --------------------------------------------------------

    empty_documents = 0
    missing_metadata = 0
    duplicate_ids = 0

    seen_ids = set()

    missing_by_field = {
        field: 0 for field in REQUIRED_METADATA
    }

    years = {}
    sources = {}

    offset = 0

    while offset < total:

        batch = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas"],
        )

        ids = batch.get("ids", [])
        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []

        if not ids:
            break

        for chunk_id, document, metadata in zip(
            ids, documents, metadatas
        ):

            # Vérification des identifiants
            if chunk_id in seen_ids:
                duplicate_ids += 1

            seen_ids.add(chunk_id)

            # Vérification des textes
            if not document or not document.strip():
                empty_documents += 1

            # Vérification des métadonnées
            metadata = metadata or {}

            missing_fields = [
                field
                for field in REQUIRED_METADATA
                if metadata.get(field) is None
                or str(metadata.get(field)).strip() == ""
            ]

            if missing_fields:
                missing_metadata += 1

            for field in missing_fields:
                missing_by_field[field] += 1

            # Répartition par année
            year = metadata.get("year")

            if year is not None:
                year = str(year)
                years[year] = years.get(year, 0) + 1

            # Répartition par source
            source = metadata.get("source")

            if source is not None:
                source = str(source)
                sources[source] = sources.get(source, 0) + 1

        offset += len(ids)

    print("\n--- QUALITÉ DES DONNÉES ---")
    print(f"Documents vides : {empty_documents}")
    print(
        "Chunks avec métadonnées manquantes : "
        f"{missing_metadata}"
    )
    print(f"Identifiants dupliqués : {duplicate_ids}")

    print("\n--- MÉTADONNÉES MANQUANTES PAR CHAMP ---")

    for field, count in missing_by_field.items():
        print(f"{field} : {count}")

    print("\n--- RÉPARTITION PAR ANNÉE ---")

    for year, count in sorted(years.items()):
        print(f"{year} : {count} chunks")

    print("\n--- RÉPARTITION PAR SOURCE ---")

    for source, count in sorted(sources.items()):
        print(f"{source} : {count} chunks")

    # --------------------------------------------------------
    # 4. TEST DE RECHERCHE SÉMANTIQUE AVEC E5
    # --------------------------------------------------------

    print("\n--- TEST DE RECHERCHE SÉMANTIQUE E5 ---")
    print(f"Question : {TEST_QUERY}")

    if total == 0:
        print("Recherche impossible : collection vide.")
        return

    # E5 exige le préfixe query: pour les questions.
    query_text = QUERY_PREFIX + TEST_QUERY

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(5, total),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    result_ids = result.get("ids", [[]])[0]
    result_docs = result.get("documents", [[]])[0]
    result_metas = result.get("metadatas", [[]])[0]
    result_distances = result.get("distances", [[]])[0]

    if not result_ids:
        print("Aucun résultat retourné.")
        return

    for i, chunk_id in enumerate(result_ids):

        metadata = result_metas[i] or {}
        document = result_docs[i] or ""
        distance = result_distances[i]

        print(f"\nRésultat {i + 1}")
        print(f"ID : {chunk_id}")
        print(f"Source : {metadata.get('source')}")
        print(f"Année : {metadata.get('year')}")
        print(f"Page : {metadata.get('page')}")
        print(f"Index du chunk : {metadata.get('chunk_index')}")
        print(f"Distance cosinus : {distance:.4f}")
        print(f"Extrait : {document[:350]}")


# ============================================================
# 5. FONCTION PRINCIPALE
# ============================================================

def main():

    print("=" * 60)
    print("AUDIT DE LA BASE VECTORIELLE CHROMADB - E5")
    print("=" * 60)

    print(f"\nChemin : {CHROMA_PATH}")

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"Base ChromaDB introuvable : {CHROMA_PATH}"
        )

    # Connexion à la base existante
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    # Vérifier que la collection E5 existe
    collections = client.list_collections()

    collection_names = [
        item.name if hasattr(item, "name") else str(item)
        for item in collections
    ]

    print(f"\nCollections trouvées : {collection_names}")

    if COLLECTION_NAME not in collection_names:
        raise ValueError(
            f"Collection introuvable : {COLLECTION_NAME}"
        )

    # Chargement du modèle E5
    print(f"\nChargement du modèle : {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # Charger uniquement la collection E5
    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    audit_collection(collection, model)

    print("\n" + "=" * 60)
    print("AUDIT TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()
