from langchain_community.document_loaders import UnstructuredWordDocumentLoader


doc_loader = UnstructuredWordDocumentLoader(
    file_path="assets/alibaba-more.docx",
    mode="elements"
)

doc = doc_loader.load()
print(doc)
print(len(doc))
