import os
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 使用项目统一的 aliQwen-api；DashScopeEmbeddings 默认只读 DASHSCOPE_API_KEY，故显式传入
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("aliQwen-api"),
)

text = "This is a test document."

# 单条文本 → 一个向量（列表）；这类写法更贴近“把用户问题转成查询向量”
query_result = embeddings.embed_query(text)
# sep=""：print 多个参数时用空字符串连接，默认是空格；这里让「文本向量长度：」和数字紧挨着输出，中间不留空
print("文本向量长度：", len(query_result), sep="")

# 多条文本 → 多个向量（列表的列表）；这类写法更贴近“批量建索引”
doc_results = embeddings.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!",
    ]
)
print(doc_results)
# sep=""：多个参数之间不加空格，输出如「文本向量数量：5，文本向量长度：1024」
print(
    "文本向量数量：", len(doc_results), "，文本向量长度：", len(doc_results[0]), sep=""
)
