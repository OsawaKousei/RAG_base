import os

import torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create cache directory
os.makedirs("./pretrained_models", exist_ok=True)

# Load the model and tokenizer
model_name = "llm-book/Swallow-7b-hf-oasst1-21k-ja"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir="./pretrained_models",
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./pretrained_models")

embedding_model_name = "BAAI/bge-m3"
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    cache_folder="./pretrained_models",
    model_kwargs={"model_kwargs": {"torch_dtype": torch.float16}},
)
