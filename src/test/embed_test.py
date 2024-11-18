import torch
from langchain_huggingface import HuggingFaceEmbeddings

# モデルの読み込み
embedding_model_name = "BAAI/bge-m3"

embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"model_kwargs": {"torch_dtype": torch.float16}},
)

sample_texts = [
    "日本で一番高い山はなんですか？",
    "日本で一番高い山は富士山です。",
]

sample_embeddings = embedding_model.embed_documents(sample_texts)
print(sample_embeddings)

similarity = torch.nn.functional.cosine_similarity(
    torch.tensor([sample_embeddings[0]]),
    torch.tensor([sample_embeddings[1]]),
)

print(similarity)
