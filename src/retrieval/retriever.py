
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

MODEL_NAME = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "bank_reports_e5"

QUERY_PREFIX = "query: "

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


STOPWORDS_FR = {
    "a", "au", "aux",
    "avec", "ce", "ces", "dans",
    "de", "des", "du",
    "elle", "en", "et", "est",
    "eux", "il", "ils",
    "je", "la", "le", "les",
    "leur", "lui",
    "ma", "mais", "me", "meme",
    "mes", "moi", "mon",
    "ne", "nos", "notre",
    "nous", "on", "ou",
    "par", "pas", "pour",
    "qu", "que", "quel", "quelle",
    "quelles", "quels",
    "qui", "sa", "se",
    "ses", "son", "sur",
    "ta", "te", "tes",
    "toi", "ton", "tu",
    "un", "une",
    "vos", "votre", "vous",
    "y", "d"
}


def tokenize(text: str) -> list[str]:
    """Normalise et tokenise le texte."""

    tokens = re.findall(
        r"\b\w+\b",
        normalize(text)
    )

    return [
        token
        for token in tokens
        if token not in STOPWORDS_FR
        and not re.fullmatch(r"(19|20)\d{2}", token)
    ]


def expand_query(question: str) -> list[str]:
    """Ajoute des synonymes métier aux termes de la question."""

    tokens = tokenize(question)

    synonyms = {
        "croissance": [
            "augmentation",
            "accroissement",
            "progression",
            "hausse",
        ],
        "depots": [
            "collecte",
            "encours",
        ],
        "taux": [
            "pourcentage",
        ],
    }

    expanded_tokens = list(tokens)

    for token in tokens:
        expanded_tokens.extend(
            synonyms.get(token, [])
        )

    return list(dict.fromkeys(expanded_tokens))


def build_expanded_question(question: str) -> str:
    """Construit une question enrichie avec des synonymes métier."""

    expanded_tokens = expand_query(question)

    return question + " " + " ".join(expanded_tokens)


def reformulate_vector_question(question: str) -> str:

    normalized = normalize(question)

    if (
        "depot" in normalized
        and "investissement" in normalized
    ):
        return (
            "Quel est le pourcentage d augmentation "
            "de l encours des dépôts d investissement ?"
        )

    return question


# ============================================================
# 3. RETRIEVER HYBRIDE
# ============================================================

class BankReportRetriever:

    def __init__(self):

        print("Chargement du modèle E5-Base...")

        self.model = SentenceTransformer(MODEL_NAME)

        print("Connexion à ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        print(f"Collection chargée : {COLLECTION_NAME}")

        print(
            f"Nombre de chunks : {self.collection.count()}"
        )

    # --------------------------------------------------------
    # 3.1 Recherche vectorielle E5
    # --------------------------------------------------------

    def _vector_search(
        self,
        question: str,
        top_k: int,
        year: str | None,
        use_reformulation: bool = True,
    ) -> dict:

        total = self.collection.count()

        if total == 0:
            return {}

        where = {"year": str(year)} if year else None

        if use_reformulation:
            query = reformulate_vector_question(question)
        else:
            query = question

        query_text = QUERY_PREFIX + query

        query_embedding = self.model.encode(
            [query_text],
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
                "id": chunk_id,
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
    # 3.2 Recherche lexicale BM25
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

        query_tokens = expand_query(question)

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

        normalized_question = normalize(question)

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
    # 3.3 Recherche hybride + fusion RRF
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

        candidate_k = min(
            max(top_k * 10, 50),
            total,
        )

        candidates = self._vector_search(
            question=question,
            top_k=candidate_k,
            year=year,
        )

        lexical_results = self._lexical_search(
            question=question,
            top_k=candidate_k,
            year=year,
        )

        for lexical_rank, result in enumerate(
            lexical_results,
            start=1,
        ):

            chunk_id = result["id"]

            if chunk_id not in candidates:

                candidates[chunk_id] = {
                    "id": chunk_id,
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

            if asks_for_pnb and (
                "produit net bancaire" in normalized_text
                or "pnb" in tokenize(text)
            ):
                item["rrf_score"] += 0.005

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

            if asks_for_net_result and (
                "resultat net" in normalized_text
            ):
                item["rrf_score"] += 0.005

        ranked = sorted(
            candidates.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        passages = []

        for item in ranked[:top_k]:

            metadata = item["metadata"]

            passages.append({
                "chunk_id": item["id"],
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
            print(f"Chunk ID : {result['chunk_id']}")
            print(f"Source : {result['source']}")
            print(f"Année : {result['year']}")
            print(f"Page : {result['page']}")
            print(f"Rang vectoriel : {result['vector_rank']}")
            print(f"Rang lexical : {result['lexical_rank']}")
            print(f"Score RRF : {result['rrf_score']}")
            print(f"Texte : {result['text'][:500]}")
