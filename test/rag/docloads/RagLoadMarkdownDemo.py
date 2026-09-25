from langchain_community.document_loaders import UnstructuredMarkdownLoader

doc = UnstructuredMarkdownLoader(file_path="assets/sample.md", mode="elements").load()
print(doc)
print(type(doc))
print(len(doc))