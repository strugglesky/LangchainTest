import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Redis
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# 1. 初始化嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3", dashscope_api_key=os.getenv("aliQwen-api")
)


