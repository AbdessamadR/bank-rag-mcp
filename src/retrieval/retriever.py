
from pathlib import Path
import math
import re
import unicodedata
from collections import Counter

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "data" / "vectorstore" / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "bank_reports"

RRF_CONSTANT = 60
BM25_K1 = 1.5
BM25_B = 0.75


# ============================================================
# 2. NORMALISATION ET TOKENISATION
# ============================================================

def normalize(text: str) -> str:
    """Normalise le texte : minuscules et suppression des accents."""
    text = unicodedata.normalize("NFD", text.lower())

    return "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )


def tokenize(text: str) -> list[str]:
    """Découpe le texte normalisé en tokens."""
    return re.findall(r"\b\w+\b", normalize(text))


# ============================================================
# 3. RETRIEVER HYBRIDE
# ============================================================

class BankReportRetriever:

    def __init__(self):
        print("Chargement du modèle d'embeddings...")

        self.model = SentenceTransformer(MODEL_NAME)

        print("Connexion à ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        print(
            f"Base vectorielle chargée : "
            f"{self.collection.count()} chunks"
        )

    # --------------------------------------------------------
    # Recherche vectorielle
    # --------------------------------------------------------

    def _vector_search(
        self,
        question: str,
        top_k: int,
        year: str | None,
    ) -> dict:

        total = self.collection.count()

        if total == 0:
            return {}

        where = {"year": str(year)} if year else None

        query_embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(top_k, total),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where:
            params["where"] = where

        results = self.collection.query(**params)

        candidates = {}

        for rank, chunk_id in enumerate(
            results["ids"][0],
            start=1,
        ):
            candidates[chunk_id] = {
                "text": results["documents"][0][rank - 1] or "",
                "metadata": (
                    results["metadatas"][0][rank - 1] or {}
                ),
                "distance": results["distances"][0][rank - 1],
                "vector_rank": rank,
                "lexical_rank": None,
                "rrf_score": 1 / (RRF_CONSTANT + rank),
            }

        return candidates

    # --------------------------------------------------------
    # Recherche lexicale BM25
    # --------------------------------------------------------

    def _lexical_search(
        self,
        question: str,
        top_k: int,
        year: str | None,
    ) -> list[dict]:

        where = {"year": str(year)} if year else None

        params = {
            "include": [
                "documents",
                "metadatas",
            ],
        }

        if where:
            params["where"] = where

        data = self.collection.get(**params)

        documents = data["documents"]
        ids = data["ids"]
        metadatas = data["metadatas"]

        if not documents:
            return []

        query_tokens = tokenize(question)

        if not query_tokens:
            return []

        tokenized_docs = [
            tokenize(document or "")
            for document in documents
        ]

        doc_count = len(tokenized_docs)

        doc_lengths = [
            len(tokens)
            for tokens in tokenized_docs
        ]

        avg_length = (
            sum(doc_lengths) / doc_count
            if doc_count
            else 1
        )

        document_frequencies = Counter()

        for tokens in tokenized_docs:
            document_frequencies.update(set(tokens))

        query_frequencies = Counter(query_tokens)
        scored_documents = []

        for index, tokens in enumerate(tokenized_docs):

            if not tokens:
                continue

            frequencies = Counter(tokens)
            doc_length = doc_lengths[index]
            score = 0.0

            for term, query_frequency in query_frequencies.items():

                tf = frequencies.get(term, 0)

                if tf == 0:
                    continue

                df = document_frequencies[term]

                idf = math.log(
                    1
                    + (doc_count - df + 0.5)
                    / (df + 0.5)
                )

                denominator = tf + BM25_K1 * (
                    1
                    - BM25_B
                    + BM25_B * doc_length / (avg_length or 1)
                )

                score += (
                    idf
                    * tf
                    * (BM25_K1 + 1)
                    / denominator
                ) * min(query_frequency, 2)

            # Petit bonus pour l'expression exacte.
            normalized_question = normalize(question)
            normalized_document = normalize(
                documents[index] or ""
            )

            if normalized_question in normalized_document:
                score += 2.0

            if score > 0:
                scored_documents.append({
                    "id": ids[index],
                    "score": score,
                    "text": documents[index] or "",
                    "metadata": metadatas[index] or {},
                })

        scored_documents.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return scored_documents[:top_k]

    # --------------------------------------------------------
    # Recherche hybride + fusion RRF
    # --------------------------------------------------------

    def search(
        self,
        question: str,
        top_k: int = 5,
        year: str | None = None,
    ) -> list[dict]:

        question = question.strip()

        if not question:
            raise ValueError(
                "La question ne peut pas être vide."
            )

        if top_k < 1:
            raise ValueError(
                "top_k doit être supérieur ou égal à 1."
            )

        if year is not None:
            year = str(year)

        total = self.collection.count()

        if total == 0:
            return []

        # Recherche plus large que le nombre final demandé.
        candidate_k = min(max(top_k * 10, 50), total)

        # 1. Recherche vectorielle.
        candidates = self._vector_search(
            question=question,
            top_k=candidate_k,
            year=year,
        )

        # 2. Recherche lexicale.
        lexical_results = self._lexical_search(
            question=question,
            top_k=candidate_k,
            year=year,
        )

        # 3. Fusion des classements.
        for lexical_rank, result in enumerate(
            lexical_results,
            start=1,
        ):

            chunk_id = result["id"]

            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "distance": None,
                    "vector_rank": None,
                    "lexical_rank": lexical_rank,
                    "rrf_score": 0.0,
                }
            else:
                candidates[chunk_id]["lexical_rank"] = (
                    lexical_rank
                )

            candidates[chunk_id]["rrf_score"] += (
                1 / (RRF_CONSTANT + lexical_rank)
            )

        # 4. Reclassement par expressions importantes.
        normalized_question = normalize(question)

        asks_for_pnb = (
            "produit net bancaire" in normalized_question
            or "pnb" in tokenize(question)
        )

        asks_for_consolidated = any(
            term in normalized_question
            for term in [
                "consolide",
                "consolidee",
                "consolidation",
                "du groupe",
            ]
        )

        asks_for_net_result = (
            "resultat net" in normalized_question
        )

        for item in candidates.values():

            text = item["text"] or ""
            normalized_text = normalize(text)

            # Bonus pour la présence du terme PNB.
            if asks_for_pnb and (
                "produit net bancaire" in normalized_text
                or "pnb" in tokenize(text)
            ):
                item["rrf_score"] += 0.005

            # Bonus lorsque la question vise le consolidé.
            if asks_for_consolidated and any(
                phrase in normalized_text
                for phrase in [
                    "resultat net consolide",
                    "produit net bancaire consolide",
                    "resultats consolides du groupe",
                    "pnb consolide",
                ]
            ):
                item["rrf_score"] += 0.015

            # Bonus pour le sujet exact du résultat net.
            if asks_for_net_result and (
                "resultat net" in normalized_text
            ):
                item["rrf_score"] += 0.005

        # 5. Tri final.
        ranked = sorted(
            candidates.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        # 6. Préparation des résultats.
        passages = []

        for item in ranked[:top_k]:

            metadata = item["metadata"]

            passages.append({
                "text": item["text"],
                "source": metadata.get("source"),
                "year": metadata.get("year"),
                "page": metadata.get("page"),
                "chunk_index": metadata.get("chunk_index"),
                "distance": (
                    round(item["distance"], 4)
                    if item["distance"] is not None
                    else None
                ),
                "vector_rank": item["vector_rank"],
                "lexical_rank": item["lexical_rank"],
                "rrf_score": round(
                    item["rrf_score"],
                    6,
                ),
            })

        return passages


# ============================================================
# 4. TESTS MANUELS
# ============================================================

if __name__ == "__main__":

    retriever = BankReportRetriever()

    questions = [
        "produit net bancaire",
        "Quel est le produit net bancaire consolidé en 2024 ?",
        "Quelle est la progression du produit net bancaire entre 2023 et 2024 ?",
        "Quel est le résultat net consolidé en 2024 ?",
    ]

    for question in questions:

        print("\n" + "=" * 80)
        print(f"QUESTION : {question}")

        results = retriever.search(
            question=question,
            top_k=5,
            year="2024",
        )

        if not results:
            print("Aucun résultat trouvé.")
            continue

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(f"\n--- Résultat {rank} ---")
            print(f"Source : {result['source']}")
            print(f"Année : {result['year']}")
            print(f"Page : {result['page']}")
            print(
                f"Rang vectoriel : "
                f"{result['vector_rank']}"
            )
            print(
                f"Rang lexical : "
                f"{result['lexical_rank']}"
            )
            print(
                f"Score RRF : "
                f"{result['rrf_score']}"
            )
            print(f"Texte : {result['text'][:500]}")
