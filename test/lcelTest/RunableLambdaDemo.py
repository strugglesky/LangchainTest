import os

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from loguru import logger

from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    temperature=0.0,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def debug_print(x):
    """打印中间结果，并把文本包装成 chain2 所需的 {"input": 文本} 结构。"""
    logger.info(f"中间结果:{x}")
    return {"input": x}


# 子链 1：中文介绍某主题，输出 str
prompt1 = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个知识渊博的计算机专家，请用中文简短回答"),
        ("human", "请简短介绍什么是{topic}"),
    ]
)
parser1 = StrOutputParser()
chain1 = prompt1 | model | parser1

# 子链 2：将 input 翻译成英文
prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个翻译助手，将用户输入内容翻译成英文"),
        ("human", "{input}")
    ]
)
parser2 = StrOutputParser()
chain2 = prompt2 | model | parser2

full_chain = chain1 | RunnableLambda(debug_print) | chain2
result = full_chain.invoke({"topic": "langchain"})
print(result)
print(type(result))


