
from src.retrieval.retriever import BankReportRetriever


retriever = BankReportRetriever()

tests = [
    {
        "question": "Quel est le produit net bancaire consolidé en 2024 ?",
        "year": "2024",
        "expected_page": 35,
        "expected_phrase": "34,5 milliards de dirhams",
    },
    {
        "question": "Quelle est la progression du produit net bancaire consolidé entre 2023 et 2024 ?",
        "year": "2024",
        "expected_page": 35,
        "expected_phrase": "amélioration de 15,2%",
    },
    {
        "question": "Quel est le résultat net consolidé en 2024 ?",
        "year": "2024",
        "expected_page": 35,
        "expected_phrase": "11,7 milliards de dirhams",
    },
]

total_hits = 0

for test in tests:
    results = retriever.search(
        question=test["question"],
        top_k=5,
        year=test["year"],
    )

    expected_phrase = test["expected_phrase"].casefold()

    matching_ranks = [
        rank
        for rank, result in enumerate(results, start=1)
        if result["page"] == test["expected_page"]
        and expected_phrase in result["text"].casefold()
    ]

    print("\n" + "=" * 75)
    print("Question :", test["question"])

    if matching_ranks:
        rank = matching_ranks[0]
        print(f"PASS : passage trouvé au rang {rank}")
        total_hits += 1
    else:
        print("À AMÉLIORER : passage attendu absent du top 5")

    for rank, result in enumerate(results, start=1):
        print(
            f"  Rang {rank} | Page {result['page']} "
            f"| Score RRF {result['rrf_score']}"
        )

print("\n" + "=" * 75)
print(f"Résultat global : {total_hits}/{len(tests)} passages retrouvés")
