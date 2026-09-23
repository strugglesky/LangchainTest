import os
import dashscope
from http import HTTPStatus
from dotenv import load_dotenv

load_dotenv()
dashscope.api_key = os.getenv("aliQwen-api")

# 待向量化的单句文本
input_text = "衣服的质量杠杠的"

# 调用百炼文本嵌入接口：先看清“请求长什么样、返回结构长什么样”
resp = dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    input=input_text,
)

if resp.status_code == HTTPStatus.OK:
    # 这里直接打印完整响应，是为了先观察响应结构；后续案例再逐步只取 embedding 向量使用
    vector = resp['output']['embeddings'][0]['embedding']
    print(vector)
    print(len(vector))

