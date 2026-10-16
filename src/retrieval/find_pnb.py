
from pathlib import Path

import chromadb


ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "data" / "vectorstore" / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection(name="bank_reports")

keywords = [
    "produit net bancaire",
    "PNB",
    "produit net bancaire consolidé",
]

total = collection.count()
batch_size = 200
matches = []

for offset in range(0, total, batch_size):
    batch = collection.get(
        limit=batch_size,
        offset=offset,
        include=["documents", "metadatas"],
    )

    for i, document in enumerate(batch["documents"]):
        metadata = batch["metadatas"][i]

        if str(metadata.get("year")) != "2024":
            continue

        text_lower = document.lower()

        if any(keyword.lower() in text_lower for keyword in keywords):
            matches.append({
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "chunk_index": metadata.get("chunk_index"),
                "text": document,
            })

print(f"Nombre de passages trouvés : {len(matches)}")

for i, item in enumerate(matches, start=1):
    print("\n" + "=" * 80)
    print(f"Passage {i}")
    print(f"Source : {item['source']}")
    print(f"Page : {item['page']}")
    print(f"Chunk : {item['chunk_index']}")
    print(f"Texte : {item['text'][:1200]}")
