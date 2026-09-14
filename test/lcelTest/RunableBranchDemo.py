import os

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from loguru import logger


from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

# 英语分支：提示词模板 + 占位符 query
english_prompt = ChatPromptTemplate.from_messages(
    [("system", "你是一个英语翻译专家，你叫小英"), ("human", "{query}")]
)

japanese_prompt = ChatPromptTemplate.from_messages(
    [("system", "你是一个日语翻译专家，你叫小日"), ("human", "{query}")]
)

korean_prompt = ChatPromptTemplate.from_messages(
    [("system", "你是一个韩语翻译专家，你叫小韩"), ("human", "{query}")]
)

test_queries = [
    {"query": '请你用韩语翻译这句话:"见到你很高兴"'},
    {"query": '请你用日语翻译这句话:"见到你很高兴"'},
    {"query": '请你用英语翻译这句话:"见到你很高兴"'},
]

def determine_language(inputs):
    """根据 query 中的关键词判断语言类型，供分支条件使用。"""
    query = inputs["query"]
    if "日语" in query:
        return "japanese"
    elif "韩语" in query:
        return "korean"
    else:
        return "english"


model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
parser = StrOutputParser()

chain = RunnableBranch(
    (lambda x: determine_language(x) == 'japanese', japanese_prompt | model | parser),
    (lambda x: determine_language(x) == 'korean', korean_prompt | model | parser),
    (english_prompt | model | parser),
)


for query_input in test_queries:
    language = determine_language(query_input)
    logger.info(f"Language: {language}")

    if language == "japanese":
        chatPromptTemplate = japanese_prompt
    elif language == "korean":
        chatPromptTemplate = korean_prompt
    else:
        chatPromptTemplate = english_prompt

    # 仅作演示：格式化后的提示词内容（实际执行时由 chain.invoke 内部完成）
    formatted_messages = chatPromptTemplate.format_messages(**query_input)
    logger.info("格式化后的提示词:")
    for msg in formatted_messages:
        logger.info(f"[{msg.type}]: {msg.content}")

    # 一次 invoke：Branch 会根据 query 自动选分支并执行对应子链
    result = chain.invoke(query_input)
    logger.info(f"输出结果: {result}\n")
