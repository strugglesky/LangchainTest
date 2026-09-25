from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader(file_path="assets/sample.pdf", extraction_mode="plain")

docs = pdf_loader.load()
print(docs)
print(len(docs))
