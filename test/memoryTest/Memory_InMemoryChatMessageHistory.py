from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from loguru import logger
import os

llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 创建内存版历史实例（BaseChatMessageHistory 的实现）
history = InMemoryChatMessageHistory()

history.add_user_message("你好 我是张三")
ai_message = llm.invoke(history.messages)
logger.info(f"第一次回答\n{ai_message.content}")
# 手动把 AI 回复写回 history；否则下一轮只会看到用户消息，达不到“多轮记忆”的效果
history.add_message(ai_message)

history.add_user_message("我叫什么？")
llm.invoke(history.messages)
