from langchain_community.document_loaders import TextLoader

file_path = "assets/sample.txt"
encoding = "utf-8"

# load() 为 BaseLoader 统一接口，返回 List[Document]
docs = TextLoader(file_path, encoding).load()

print(docs)
print(type(docs))