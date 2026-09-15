from cgitb import strong

from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import os

llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 按 session_id 保存多份历史，便于多用户/多会话；生产可改为 Redis 等
store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个友好的中文助理，会根据上下文回答问题。"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm | StrOutputParser()

with_history_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

cfg_user_001 = {"configurable": {"session_id": "user-001"}}
cfg_user_002 = {"configurable": {"session_id": "user-002"}}

res1 = with_history_chain.invoke({"question": "我是张三"}, cfg_user_001)
print(res1)

res2 = with_history_chain.invoke({"question": "我叫什么？"}, cfg_user_001)
print(res2)

res3 = with_history_chain.invoke({"question": "我是斯里"}, cfg_user_002)
print(res3)

res4 = with_history_chain.invoke({"question": "我叫什么？"}, cfg_user_002)
print(res4)

print("\n===== 所有会话历史 =====")
for session_id, chat_history in store.items():
    print(f"\n会话 ID: {session_id}")
    for message in chat_history.messages:
        print(f"[{message.type}] {message.content}")


