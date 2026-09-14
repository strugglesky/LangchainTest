from langchain.chat_models import init_chat_model
import os
from typing import TypedDict, Annotated

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv(encoding="utf-8")

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


messages = [{"role": "user", "content": "任意生成三种动物，以及他们的 emoji 表情"}]

class Animal(TypedDict):
    animal: Annotated[str, "动物的名称"]
    emoji: Annotated[str, "动物的 emoji 表情"]

class AnimalList(TypedDict):
    animals: Annotated[list[Animal], "动物与表情列表"]


structured_model = model.with_structured_output(
    AnimalList
)
result = structured_model.invoke(messages)

print(result)
print(type(result))
print(type(result["animals"]))
