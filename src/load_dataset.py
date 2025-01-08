from datasets import load_dataset

dataset = load_dataset(
    "llm-book/aio-retriever", trust_remote_code=True, cache_dir="./data"
)

print(dataset)
