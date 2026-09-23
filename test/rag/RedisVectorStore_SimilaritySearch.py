from langchain_redis import RedisConfig, RedisVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embeddingModel = DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=os.getenv("aliQwen-api"))

redis_vector_store = RedisVectorStore(embeddings=embeddingModel,
                         config=RedisConfig(index_name="newsgroups", redis_url="redis://localhost:6379"))

query = "我喜欢用什么手机"
results = redis_vector_store.similarity_search_with_score(query, k=3)

print("=== 查询结果 ===")
for i, (doc, score) in enumerate(results, 1):
    # 这里把“距离”近似换算成“相似度”只是为了展示更直观；工程里请以具体返回定义为准
    similarity = 1 - score
    print(f"结果 {i}:")
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
    print(f"相似度: {similarity:.4f}")

