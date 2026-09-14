from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个{role}，请简短回答我提出的问题"),
        ("human", "请回答:{question}"),
    ]
)

# 使用 invoke 渲染提示词，返回 PromptValue，可直接交给模型（统一接口）
prompt = chat_prompt.invoke(
    {"role": "AI助手", "question": "什么是LangChain，简洁回答100字以内"}
)
logger.info(prompt)

# 初始化聊天模型（同样实现 Runnable，支持 invoke/stream/batch）
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 模型接收上一步的 PromptValue，返回 AIMessage
result = model.invoke(prompt)
logger.info(f"********>模型原始输出:\n{result}")

# 字符串输出解析器（Runnable）：从 AIMessage 中取出文本，得到更适合业务继续处理的文本结果
parser = StrOutputParser()

# 解析器接收 AIMessage，这里得到的是文本结果
parse = parser
response = parse.invoke(result)
logger.info(f"解析后的结构化结果:\n{response}")
logger.info(f"结果类型: {type(response)}")


chain = chat_prompt | model | parse
print(type(chain))
print(isinstance(chain, RunnableSequence))
result_chain = chain.invoke({"role": "AI助手", "question": "什么是LangChain，简洁回答100字以内"})
logger.info(f"Chain执行结果:\n{result_chain}")
logger.info(f"Chain执行结果类型: {type(result_chain)}")


