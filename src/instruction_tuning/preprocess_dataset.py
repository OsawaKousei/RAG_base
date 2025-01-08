from pprint import pprint
from typing import Any

from datasets import load_dataset

dataset = load_dataset(
    "llm-book/aio-retriever", trust_remote_code=True, cache_dir="./data"
)

if __name__ == "__main__":
    print(dataset)


def filter_example(example: dict[str, Any], max_passages: int = 3) -> bool:
    if len(example["positive_passage_indices"]) == 0:
        return False
    if example["positive_passage_indices"][0] >= max_passages:
        return False

    return True


dataset = dataset.filter(filter_example)


def process_example(example: dict[str, Any], max_passages: int = 3) -> dict[str, Any]:
    question = example["question"]
    answer = example["answers"][0]
    passages = [p["text"] for p in example["passages"]]

    passages = passages[:max_passages]
    messages: list[dict[str, Any]] = []
    prompt_text = "".join(
        [
            "あなたは今からクイズに答えてもらいます。",
            "問題を与えますので、その回答のみを簡潔に出してください。\n",
            "また解答の参考になりうるテキストを与えます。",
            "解答を含まない場合もありますので、その場合は無視して下さい・\n\n",
            "---\n",
            "---\n".join(passages),
            "\n---\n\n",
            f"問題: {question}\n",
        ]
    )
    messages.append({"role": "user", "content": prompt_text})
    messages.append({"role": "assistant", "content": answer})

    example["messages"] = messages
    return example


dataset = dataset.map(process_example, remove_columns=dataset["train"].column_names)

if __name__ == "__main__":
    print(dataset)
    pprint(dataset["validation"][0])
