import dashscope
import json
import os
from http import HTTPStatus
from dotenv import load_dotenv

load_dotenv()
# 多模态 call 内部用 get_default_api_key()，必须提前设置 dashscope.api_key，否则报 No api key provided
dashscope.api_key = os.getenv("aliQwen-api")

# 调用多模态 embedding 接口：支持文本或图像输入，本例只保留最小的文本演示
resp = dashscope.MultiModalEmbedding.call(
    model="tongyi-embedding-vision-plus",
    input=[{"image": "https://pic3.zhimg.com/v2-d1f733345b0d11ea4d1bde0e2511dbc8_720w.jpg?source=172ae18b"}],
)

result = ""

if resp.status_code == HTTPStatus.OK:
    result = {
        "status_code": resp.status_code,
        "request_id": getattr(resp, "request_id", ""),
        "code": getattr(resp, "code", ""),
        "message": getattr(resp, "message", ""),
        "output": resp.output,
        "usage": resp.usage,
    }
    # ensure_ascii=False：中文等非 ASCII 按原样输出，不转成 \uxxxx；indent=4：每层缩进 4 格，便于阅读
    print(json.dumps(result, ensure_ascii=False, indent=4))

print("=================================")
print()

# 从完整结果中取出第一条 embedding 向量；后续若要做相似度比较，可直接使用这组数值
embedding_values = result["output"]["embeddings"][0]["embedding"]
print(json.dumps(embedding_values, ensure_ascii=False))