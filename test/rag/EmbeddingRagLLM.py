from langchain.chat_models import init_chat_model
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.prompts import PromptTemplate
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Redis
from dotenv import load_dotenv

load_dotenv()

# 大模型：用于最终根据「检索到的上下文 + 用户问题」生成回答
llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

prompt_template = """
    请使用以下提供的文本内容来回答问题。仅使用提供的文本信息，
    如果文本中没有相关信息，请回答"抱歉，提供的文本中没有这个信息"。

    文本内容：
    {context}

    问题：{question}

    回答：
    "
"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# 嵌入模型：用于文档与查询的向量化
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3", dashscope_api_key=os.getenv("aliQwen-api")
)

loader = Docx2txtLoader(file_path="alibaba-java.docx")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, length_function=len
)

split_documents = text_splitter.split_documents(documents)
print(f'分割后的文档数量：{len(split_documents)}')

vector_store = Redis.from_documents(
    documents=split_documents,
    embedding=embeddings,
    redis_url="redis://localhost:6379",
    index_name="my_index3",
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

def print_RagContent(x: dict[str, list]) -> dict:
    print(x["context"][0].page_content)
    return x

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | RunnableLambda(print_RagContent) | prompt | llm

result = rag_chain.invoke("00000和A0001分别是什么意思")

print(result.content)

