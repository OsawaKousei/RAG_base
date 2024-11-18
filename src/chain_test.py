import torch
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.trainer_utils import set_seed

set_seed(42)

# モデルの読み込み
model_name = "llm-book/Swallow-7b-hf-oasst1-21k-ja"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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

chain_resp_only = prompt_template | chat_model_resp_only

if __name__ == "__main__":
    chain_resp_only_output = chain_resp_only.invoke(
        {"query": "四国地方で一番高い山は？"}
    )
    print(chain_resp_only_output)
