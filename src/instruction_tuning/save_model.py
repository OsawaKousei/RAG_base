import torch
from peft import LoraConfig, TaskType, get_peft_model
from preprocess_dataset import dataset as train_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM

base_model_name = "llm-book/Swallow-7b-hf-oasst1-21k-ja"


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

tokenized_train_dataset = [
    tokenizer.apply_chat_template(example["messages"])
    for example in train_dataset["train"]
]

bos = tokenizer.bos_token
collator = DataCollatorForCompletionOnlyLM(
    instruction_template=bos + "ユーザ：",
    response_template=bos + "アシスタント：",
    tokenizer=tokenizer,
)

peft_config = LoraConfig(
    r=128,
    lora_alpha=128,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir="./finetuned_models/rag_it_results/checkpoints",
    bf16=True,
    max_steps=100,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    data_collator=collator,
    args=training_args,
    tokenizer=tokenizer,
)


checkpoint_path = "./finetuned_models/rag_it_results/checkpoints/checkpoint-100"

trainer.train(resume_from_checkpoint=checkpoint_path)

trainer.save_model("./finetuned_models/rag_it_results/model")
