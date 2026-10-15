
import json
from pathlib import Path
from collections import defaultdict
from statistics import median

path = Path("data/processed/pages.json")

with path.open("r", encoding="utf-8") as f:
    documents = json.load(f)

by_year = defaultdict(list)

for doc in documents:
    by_year[doc["year"]].append(doc)

print("=== AUDIT DU CORPUS ===")
print(f"Total des pages : {len(documents)}")

replacement_char = "\uFFFD"

for year in sorted(by_year):
    pages = by_year[year]
    lengths = [len(p["text"]) for p in pages]

    print(f"\nAnnée : {year}")
    print(f"Nombre de pages : {len(pages)}")
    print(f"Caractères : {sum(lengths)}")
    print(f"Longueur médiane : {median(lengths):.0f}")

    short_pages = sum(n < 100 for n in lengths)
    print(f"Pages de moins de 100 caractères : {short_pages}")

    replacement_count = sum(
        replacement_char in p["text"] for p in pages
    )
    print(
        "Pages contenant le caractère de remplacement : "
        f"{replacement_count}"
    )

print("\n=== 5 PAGES LES PLUS COURTES ===")

for doc in sorted(documents, key=lambda d: len(d["text"]))[:5]:
    print(
        f"\n{doc['year']} | page {doc['page']} "
        f"| {len(doc['text'])} caractères"
    )
    print(repr(doc["text"][:250]))

print("\n=== AUDIT TERMINÉ ===")
