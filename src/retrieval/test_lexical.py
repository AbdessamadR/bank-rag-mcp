
from pathlib import Path
import re
import unicodedata
import chromadb

ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "data" / "vectorstore" / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection(name="bank_reports")

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    return text

def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[\w]+\b", normalize(text)))

question = "produit net bancaire"
query_terms = tokenize(question)

data = collection.get(
    where={"year": "2024"},
    include=["documents", "metadatas"],
)

results = []

for document, metadata in zip(
    data["documents"],
    data["metadatas"],
):
    if not document:
        continue

    document_terms = tokenize(document)
    matched_terms = query_terms & document_terms

    if matched_terms:
        score = len(matched_terms) / len(query_terms)

        # Bonus pour l'expression exacte, indépendamment
        # des accents et de la casse.
        normalized_doc = normalize(document)
        normalized_query = normalize(question)

        if normalized_query in normalized_doc:
            score += 2

        results.append({
            "score": score,
            "page": metadata.get("page"),
            "chunk_index": metadata.get("chunk_index"),
            "source": metadata.get("source"),
            "text": document,
        })

results.sort(key=lambda item: item["score"], reverse=True)

print(f"Chunks analysés : {len(data['documents'])}")
print(f"Chunks avec correspondance : {len(results)}")

for i, result in enumerate(results[:10], start=1):
    print("\n" + "=" * 70)
    print(f"Rang : {i}")
    print(f"Score lexical : {result['score']:.2f}")
    print(f"Page : {result['page']}")
    print(f"Chunk : {result['chunk_index']}")
    print(f"Source : {result['source']}")
    print(f"Texte : {result['text'][:500]}")
