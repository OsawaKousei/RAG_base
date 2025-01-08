import logging
from logging import Formatter, StreamHandler, getLogger

import torch
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ログの設定
if __name__ == "__main__":
    logger = getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
else:
    logger = getLogger("__main__").getChild(__name__)

# モデルの読み込み
model_name = "llm-book/Swallow-7b-hf-oasst1-21k-ja"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="./pretrained_models",
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./pretrained_models")

# パラメータの設定
generation_config = {
    "max_new_tokens": 128,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
}

# パイプラインの作成
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    **generation_config,
)

# llmコンポーネントの作成
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

chat_model = ChatHuggingFace(llm=llm, tokenizer=tokenizer)

# プロンプトの作成
prompt_template = ChatPromptTemplate.from_messages([("user", "{query}")])


def chat_model_resp_only_func(chat_prompt_value: ChatPromptValue) -> str:
    """chat_modelにchat_prompt_valueを入力して、応答部分のみを返す

    Args:
        chat_prompt_value (ChatPromptValue): 入力となるChatPromptValue

    Returns:
        str: chat_modelの応答部分
    """
    chat_prompt = chat_model._to_chat_prompt(chat_prompt_value.messages)
    chat_output_message = chat_model.invoke(chat_prompt_value)
    response_text = chat_output_message.content[len(chat_prompt) :]

    assert isinstance(response_text, str)
    return response_text


chat_model_resp_only = RunnableLambda(chat_model_resp_only_func)

# モデルの読み込み
embedding_model_name = "BAAI/bge-m3"
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    cache_folder="./pretrained_models",
    model_kwargs={"model_kwargs": {"torch_dtype": torch.float16}},
)


# loaderを初期化
document_loader = JSONLoader(
    file_path="data/docs.json",
    jq_schema=".text",
    json_lines=True,
)

# ドキュメントをロード
documents = document_loader.load()

logger.debug(len(documents))
logger.debug(documents[0])
logger.debug(len(documents[0].page_content))

# splitterを初期化
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
    add_start_index=True,
)

# ドキュメントを分割
split_documents = text_splitter.split_documents(documents)

logger.debug(len(split_documents))
logger.debug(split_documents[0])
logger.debug(split_documents[1])
logger.debug(len(split_documents[0].page_content))
logger.debug(len(split_documents[1].page_content))

# vectorstoreを初期化
vectorstore = FAISS.from_documents(split_documents, embedding_model)

logger.debug(vectorstore.index.ntotal)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retrieved_documents = retriever.invoke("四国地方で一番高い山は何ですか？")

logger.debug(retrieved_documents)

rag_prompt_text = (
    "以下の文書の内容を参考にして、質問に答えてください。\n\n"
    "---\n{context}\n---\n\n質問: {query}"
)

rag_prompt_template = ChatPromptTemplate.from_messages([("user", rag_prompt_text)])


def format_documents_func(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content for document in documents)


format_documents = RunnableLambda(format_documents_func)

rag_chain = (
    {
        "context": retriever | format_documents,
        "query": RunnablePassthrough(),
    }
    | rag_prompt_template
    | chat_model_resp_only
)

if __name__ == "__main__":
    rag_output = rag_chain.invoke("四国地方で一番高い山は何ですか？")
    print(rag_output)
