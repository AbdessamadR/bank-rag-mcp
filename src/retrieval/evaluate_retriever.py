
import json
import math
from pathlib import Path

from src.retrieval.retriever import BankReportRetriever


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

GROUND_TRUTH_PATH = (
    ROOT_DIR / "data" / "evaluation" / "ground_truth.json"
)

OUTPUT_PATH = (
    ROOT_DIR / "data" / "benchmarks" / "e5_only_evaluation.json"
)

TOP_K = 5


# ============================================================
# 2. CHARGEMENT ET VALIDATION DU DATASET
# ============================================================

def load_ground_truth():
    """Load and validate evaluation questions."""

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {GROUND_TRUTH_PATH}"
        )

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError(
            "Ground truth must contain a non-empty JSON list."
        )

    required_fields = ["id", "question", "relevant_chunks"]

    for index, item in enumerate(data):

        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid question at position {index}."
            )

        for field in required_fields:
            if field not in item:
                raise ValueError(
                    f"Missing field '{field}' at position {index}."
                )

        if (
            not isinstance(item["question"], str)
            or not item["question"].strip()
        ):
            raise ValueError(
                f"Empty question for {item['id']}."
            )

        if (
            not isinstance(item["relevant_chunks"], list)
            or not item["relevant_chunks"]
        ):
            raise ValueError(
                f"Invalid relevant_chunks for {item['id']}."
            )

    return data


# ============================================================
# 3. METRIQUES D'EVALUATION
# ============================================================

def calculate_hit_rate(retrieved, relevant):
    """Return 1 if at least one relevant chunk is retrieved."""

    return float(any(chunk in relevant for chunk in retrieved))


def calculate_recall(retrieved, relevant):
    """Calculate Recall@K."""

    if not relevant:
        return 0.0

    retrieved_relevant = set(retrieved) & relevant

    return len(retrieved_relevant) / len(relevant)


def calculate_mrr(retrieved, relevant):
    """Calculate Reciprocal Rank."""

    for rank, chunk_id in enumerate(retrieved, start=1):

        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def calculate_ndcg(retrieved, relevant):
    """Calculate nDCG@K using binary relevance."""

    if not relevant:
        return 0.0

    dcg = 0.0

    for rank, chunk_id in enumerate(retrieved, start=1):

        relevance = 1 if chunk_id in relevant else 0

        dcg += relevance / math.log2(rank + 1)

    ideal_relevant = min(len(relevant), TOP_K)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# 4. EVALUATION D'UNE QUESTION - E5 SEUL
# ============================================================

def evaluate_question(retriever, question_data):
    """Evaluate one question using E5 vector retrieval only."""

    question = question_data["question"]
    year = question_data.get("year")

    relevant = set(question_data["relevant_chunks"])

    vector_results = retriever._vector_search(
        question=question,
        top_k=TOP_K,
        year=year,
        use_reformulation=False
    )

    ranked_results = sorted(
        vector_results.items(),
        key=lambda item: item[1]["vector_rank"]
    )

    retrieved = [
        chunk_id
        for chunk_id, _ in ranked_results[:TOP_K]
    ]

    return {
        "id": question_data["id"],
        "question": question,
        "year": year,
        "relevant_chunks": sorted(relevant),
        "retrieved_chunks": retrieved,
        "hit_rate": calculate_hit_rate(retrieved, relevant),
        "recall": calculate_recall(retrieved, relevant),
        "mrr": calculate_mrr(retrieved, relevant),
        "ndcg": calculate_ndcg(retrieved, relevant)
    }


# ============================================================
# 5. SAUVEGARDE DES RESULTATS
# ============================================================

def save_results(evaluations, metrics):
    """Save detailed evaluation results as JSON."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "model": "intfloat/multilingual-e5-base",
        "collection": "bank_reports_e5",
        "retrieval_method": "E5 vector only",
        "top_k": TOP_K,
        "number_of_questions": len(evaluations),
        "metrics": metrics,
        "evaluations": evaluations
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nRésultats sauvegardés dans : {OUTPUT_PATH}")


# ============================================================
# 6. EVALUATION GLOBALE
# ============================================================

def main():

    print("=" * 65)
    print("          BANK RAG - E5 VECTOR ONLY EVALUATION")
    print("=" * 65)

    ground_truth = load_ground_truth()

    print(f"\nQuestions chargées : {len(ground_truth)}")
    print(f"TOP_K : {TOP_K}")

    print("\nInitialisation du retriever E5...")

    retriever = BankReportRetriever()

    evaluations = []

    for question_data in ground_truth:

        result = evaluate_question(
            retriever,
            question_data
        )

        evaluations.append(result)

        print("\n" + "-" * 65)
        print(f"ID       : {result['id']}")
        print(f"QUESTION : {result['question']}")
        print(f"YEAR     : {result['year']}")

        print(f"EXPECTED : {result['relevant_chunks']}")
        print(f"RETRIEVED: {result['retrieved_chunks']}")

        print(f"Hit Rate@{TOP_K}: {result['hit_rate']:.3f}")
        print(f"Recall@{TOP_K}  : {result['recall']:.3f}")
        print(f"MRR@{TOP_K}     : {result['mrr']:.3f}")
        print(f"nDCG@{TOP_K}    : {result['ndcg']:.3f}")

    # --------------------------------------------------------
    # MOYENNES GLOBALES
    # --------------------------------------------------------

    total = len(evaluations)

    metrics = {
        "hit_rate": sum(
            item["hit_rate"] for item in evaluations
        ) / total,

        "recall": sum(
            item["recall"] for item in evaluations
        ) / total,

        "mrr": sum(
            item["mrr"] for item in evaluations
        ) / total,

        "ndcg": sum(
            item["ndcg"] for item in evaluations
        ) / total
    }

    # --------------------------------------------------------
    # RESULTATS FINAUX
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("                 FINAL METRICS")
    print("=" * 65)

    print(f"Number of questions : {total}")
    print(f"Hit Rate@{TOP_K}       : {metrics['hit_rate']:.4f}")
    print(f"Recall@{TOP_K}         : {metrics['recall']:.4f}")
    print(f"MRR@{TOP_K}            : {metrics['mrr']:.4f}")
    print(f"nDCG@{TOP_K}           : {metrics['ndcg']:.4f}")

    print("=" * 65)

    save_results(evaluations, metrics)


if __name__ == "__main__":
    main()
