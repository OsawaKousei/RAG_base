from pprint import pprint

import torch
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.trainer_utils import set_seed

set_seed(42)

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


def q_a_with_llm(
    llm: HuggingFacePipeline, tokenizer: AutoTokenizer, message: str
) -> str:
    """llmによる質問応答"""
    llm_prompt_messages = [
        {"role": "user", "content": message},
    ]

    llm_prompt_text = tokenizer.apply_chat_template(
        llm_prompt_messages, tokenize=False, add_generation_prompt=False
    )

    print(llm_prompt_text)

    llm_output_message = llm.invoke(llm_prompt_text)
    print(llm_output_message)

    assert isinstance(llm_output_message, str)
    return llm_output_message


def q_a_with_chat_model(chat_model: ChatHuggingFace, message: str) -> str:
    """ChatHuggingFaceによる質問応答"""
    chat_message = [HumanMessage(content=message)]
    chat_prompt = chat_model._to_chat_prompt(chat_message)
    print(chat_prompt)

    chat_output_message = chat_model.invoke(chat_prompt)
    pprint(chat_output_message)

    response_text = chat_output_message.content[len(chat_prompt) :]
    print(response_text)
    assert isinstance(response_text, str)
    return response_text


if __name__ == "__main__":
    # llm componentによる質問応答
    message = "四国地方で一番高い山は？"
    # response = q_a_with_llm(llm, tokenizer, message)
    response = q_a_with_chat_model(chat_model, message)
