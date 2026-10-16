
from pathlib import Path

import chromadb

ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "data" / "vectorstore" / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection(name="bank_reports")

results = collection.get(
    where={
        "$and": [
            {"year": "2024"},
            {"page": 35},
        ]
    },
    include=["documents", "metadatas"],
)

print(f"Nombre de chunks sur la page 35 : {len(results['documents'])}")

for i, document in enumerate(results["documents"]):
    metadata = results["metadatas"][i]

    print("\n" + "=" * 70)
    print("Chunk :", metadata.get("chunk_index"))
    print("Source :", metadata.get("source"))
    print("Texte :")
    print(document)
