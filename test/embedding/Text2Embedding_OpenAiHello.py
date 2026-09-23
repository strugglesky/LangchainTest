import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

input_text = "衣服的质量杠杠的"

# 使用 OpenAI 兼容接口连接阿里百炼：调用方式仍是 OpenAI SDK，只是连接地址改成百炼的兼容网关
client = OpenAI(
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 与 OpenAI Embedding 调用方式一致：model 为百炼模型名，input 为待向量化的文本
completion = client.embeddings.create(model="text-embedding-v4", input=input_text)

print(completion.model_dump_json())