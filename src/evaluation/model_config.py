MODELS = {
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "prefix_query": "",
        "prefix_document": "",
    },
    "bge_m3": {
        "name": "BAAI/bge-m3",
        "dimension": 1024,
        "prefix_query": "",
        "prefix_document": "",
    },
    "e5_base": {
        "name": "intfloat/multilingual-e5-base",
        "dimension": 768,
        "prefix_query": "query: ",
        "prefix_document": "passage: ",
    },
}

CHUNK_FILE = "data/processed/chunks.json"
GROUND_TRUTH_FILE = "data/evaluation/ground_truth.json"

BENCHMARK_DIR = "data/benchmarks"

TOP_K = 5

BATCH_SIZE = 32