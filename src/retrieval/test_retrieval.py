
from src.retrieval.retriever import BankReportRetriever

retriever = BankReportRetriever()

questions = [
    "produit net bancaire",
    "34,5 milliards de dirhams contre 29,9 milliards en 2023",
    "Résultats consolidés du Groupe",
    "Au 31 décembre 2024 le produit net bancaire consolidé",
]

for question in questions:
    print("\n" + "=" * 80)
    print(f"QUESTION : {question}")

    results = retriever.search(
        question,
        top_k=50,
        year="2024",
    )

    found = False

    for rank, result in enumerate(results, start=1):
        text = result["text"].lower()

        if (
            result["page"] == 35
            or "34,5 milliards" in text
            or "produit net bancaire consolidé" in text
        ):
            found = True
            print(f"\nRang : {rank}")
            print(f"Page : {result['page']}")
            print(f"Chunk : {result['chunk_index']}")
            print(f"Distance : {result['distance']}")
            print(f"Texte : {result['text'][:600]}")

    if not found:
        print("Aucun passage pertinent parmi les 50 résultats.")
