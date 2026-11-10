
import csv
import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.evaluation.model_config import (
    MODELS,
    CHUNK_FILE,
    GROUND_TRUTH_FILE,
    BENCHMARK_DIR,
    TOP_K,
    BATCH_SIZE,
)


RESULT_DIR = Path("results/embedding_benchmark")
COLLECTION_NAME = "benchmark"


def load_data():
    with open(CHUNK_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    with open(GROUND_TRUTH_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    return chunks, questions


def calculate_metrics(results, questions):
    hits = []
    recalls = []
    reciprocal_ranks = []
    ndcgs = []

    for question, retrieved_ids in zip(questions, results):
        relevant = set(question["relevant_chunks"])
        retrieved = retrieved_ids[:TOP_K]

        found = relevant.intersection(retrieved)

        hits.append(1 if found else 0)
        recalls.append(len(found) / len(relevant))

        reciprocal_rank = 0
        dcg = 0

        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in relevant:
                if reciprocal_rank == 0:
                    reciprocal_rank = 1 / rank

                dcg += 1 / (
                    __import__("math").log2(rank + 1)
                )

        ideal_hits = min(len(relevant), TOP_K)

        idcg = sum(
            1 / __import__("math").log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )

        reciprocal_ranks.append(reciprocal_rank)
        ndcgs.append(dcg / idcg if idcg else 0)

    return {
        "hit_rate_at_5": sum(hits) / len(hits),
        "recall_at_5": sum(recalls) / len(recalls),
        "mrr_at_5": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "ndcg_at_5": sum(ndcgs) / len(ndcgs),
    }


def build_collection(model, chunks, model_config, storage_path):
    client = chromadb.PersistentClient(path=str(storage_path))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    documents = [
        model_config["prefix_document"] + chunk["text"]
        for chunk in chunks
    ]

    start = time.perf_counter()

    embeddings = model.encode(
        documents,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for start_index in range(0, len(chunks), BATCH_SIZE):
        end_index = start_index + BATCH_SIZE

        batch_chunks = chunks[start_index:end_index]

        collection.add(
            ids=[c["chunk_id"] for c in batch_chunks],
            documents=[c["text"] for c in batch_chunks],
            embeddings=embeddings[start_index:end_index].tolist(),
            metadatas=[
                {
                    "year": str(c["year"]),
                    "page": int(c["page"]),
                    "source": str(c["source"]),
                }
                for c in batch_chunks
            ],
        )

    build_time = time.perf_counter() - start

    return collection, build_time


def evaluate_model(model, collection, questions, model_config):
    results = []
    latencies = []

    for question in questions:
        query_text = (
            model_config["prefix_query"] + question["question"]
        )

        start = time.perf_counter()

        query_embedding = model.encode(
            query_text,
            normalize_embeddings=True,
        )

        search_result = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=TOP_K,
            include=["distances"],
        )

        latency = time.perf_counter() - start
        latencies.append(latency)

        retrieved_ids = search_result["ids"][0]
        results.append(retrieved_ids)

    metrics = calculate_metrics(results, questions)

    metrics["avg_query_latency_ms"] = (
        sum(latencies) / len(latencies) * 1000
    )

    return metrics


def get_directory_size(path):
    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file()
    ) / (1024 * 1024)


def main():
    chunks, questions = load_data()

    print(f"Chunks : {len(chunks)}")
    print(f"Questions : {len(questions)}")

    benchmark_root = Path(BENCHMARK_DIR)
    benchmark_root.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_key, config in MODELS.items():
        print(f"\n{'=' * 50}")
        print(f"Modèle : {model_key}")
        print(f"Nom : {config['name']}")
        print(f"{'=' * 50}")

        model = SentenceTransformer(config["name"])

        actual_dimension = model.get_sentence_embedding_dimension()

        if actual_dimension != config["dimension"]:
            raise ValueError(
                f"Dimension incorrecte pour {model_key}: "
                f"{actual_dimension} au lieu de {config['dimension']}"
            )

        storage_path = benchmark_root / model_key / "chroma_db"
        storage_path.mkdir(parents=True, exist_ok=True)

        collection, build_time = build_collection(
            model,
            chunks,
            config,
            storage_path,
        )

        metrics = evaluate_model(
            model,
            collection,
            questions,
            config,
        )

        metrics.update({
            "model": model_key,
            "model_name": config["name"],
            "dimension": actual_dimension,
            "build_time_seconds": round(build_time, 2),
            "storage_mb": round(
                get_directory_size(storage_path), 2
            ),
        })

        all_results.append(metrics)

        print(json.dumps(metrics, indent=4, ensure_ascii=False))

        del model

    json_path = RESULT_DIR / "benchmark_results.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    csv_path = RESULT_DIR / "benchmark_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(all_results)

    print("\nBenchmark terminé.")
    print(f"Résultats JSON : {json_path}")
    print(f"Résultats CSV : {csv_path}")


if __name__ == "__main__":
    main()
