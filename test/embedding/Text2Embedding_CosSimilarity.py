import dashscope
import os
from http import HTTPStatus
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# 准备多句文本，用于观察“语义越接近，相似度通常越高”
texts = ["我喜欢吃苹果", "苹果是我最喜欢吃的水果", "我喜欢用苹果手机"]

embeddings = []

for text in texts:
    input_data = [{"text": text}]
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        api_key=os.getenv("aliQwen-api"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        input=input_data,
    )
    if resp.status_code == HTTPStatus.OK:
        embedding = resp.output["embeddings"][0]["embedding"]
        embeddings.append(embedding)