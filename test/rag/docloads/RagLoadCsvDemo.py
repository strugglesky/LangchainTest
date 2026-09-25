from langchain_community.document_loaders import CSVLoader

# 方式一：不指定列 → 整行（所有列）拼成一条字符串作为 page_content，metadata 通常只有 source 等
docs_all = CSVLoader(file_path="assets/sample.csv").load()
print("=== 方式一：整行作为 page_content ===")
print(docs_all[0].page_content)
print(len(docs_all))

# 方式二：指定 content_columns 与 metadata_columns → 正文只取 content 列，title/author 进 metadata，便于检索时按作者/标题过滤
docs_split = CSVLoader(
    file_path="assets/sample.csv",
    metadata_columns=["title", "author"],
    content_columns=["content"],
).load()
print("=== 方式二：content 列作为正文，title/author 进 metadata ===")
print(docs_split[0].page_content)
print(len(docs_split))

