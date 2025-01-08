import torch
from datasets import Dataset
from preprocess_dataset import dataset as eval_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

base_model_name = "./finetuned_models/rag_it_results/model"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
    cache_dir="./pretrained_models",
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name, cache_dir="./pretrained_models"
)


def evaluate(
    model: PreTrainedModel, dataset: Dataset
) -> tuple[list[str], list[str], float]:
    pred_answers = []
    gold_answers = []
    num_correct = 0

    for example in tqdm(dataset):
        model_inputs = tokenizer.apply_chat_template(
            example["messages"][:-1],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        input_length = model_inputs.shape[1]

        generated_ids = model.generate(
            model_inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        pred_answer = tokenizer.batch_decode(
            generated_ids[:, input_length:], skip_special_tokens=True
        )[0]
        gold_answer = example["messages"][-1]["content"]

        if pred_answer == gold_answer:
            num_correct += 1

        pred_answers.append(pred_answer)
        gold_answers.append(gold_answer)

    accuracy = num_correct / len(dataset)
    return pred_answers, gold_answers, accuracy


if __name__ == "__main__":
    pred_answers, gold_answers, accuracy = evaluate(model, eval_dataset["validation"])
    print(f"Accuracy: {accuracy}")  # 81.48%
