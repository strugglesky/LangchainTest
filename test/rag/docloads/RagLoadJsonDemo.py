from langchain_community.document_loaders.json_loader import JSONLoader
from numba.core.cpu_options import FastMathOptions
from redis.crc import key_slot
from spacy.attrs import FLAG30

json_loader = JSONLoader(file_path="assets/sample.json", jq_schema=".", text_content=False, )

document = json_loader.load()

print(document)
print(len(document))

