from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

# 1. 加载文档得到 Document 列表
loader = UnstructuredLoader("rag.txt")
documents = loader.load()
print(len(documents))
# 2. 同一套分割参数：块 100 字符，重叠 30
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=30, length_function=len
)
split_documents = text_splitter.split_documents(documents)
print(len(split_documents))

print(f"分割文档数量：{len(split_documents)}")
for splitter_document in split_documents:
    print(f"文档片段：{splitter_document.page_content}")
    print(
        f"文档片段大小：{len(splitter_document.page_content)}, 文档元数据：{splitter_document.metadata}"
    )




