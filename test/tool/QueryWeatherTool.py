from langchain_core.tools import tool
import json
import os
import httpx
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

def get_weather(location: str) -> str:
    """
        查询指定城市的即时天气。

        参数:
            loc: 城市名称字符串。为了提高调用成功率，建议优先传英文城市名，
                 如 Beijing、Shanghai。

        返回:
            OpenWeather 当前天气接口返回的 JSON 字符串，包含气温、体感温度、
            湿度、风速、天气描述等信息。
    """
    # Step 1. 构建请求 URL（OpenWeather 当前天气接口，见教程 5.2 API 文档）
    url = "https://api.openweathermap.org/data/2.5/weather"

    # Step 2. 设置查询参数：q=城市名，appid 从环境变量读取（安全实践），units=metric 为摄氏度，lang=zh_cn 为中文描述
    params = {
        "q": location,
        "appid": os.getenv(
            "OPENWEATHER_API_KEY"
        ),  # 从 .env 读取，勿将 Key 写死在代码中
        "units": "metric",  # 温度单位：metric=摄氏度
        "lang": "zh_cn",  # 天气描述语言：简体中文
    }

    # Step 3. 发送 GET 请求；httpx 与 requests 用法类似，timeout 避免长时间阻塞
    response = httpx.get(url, params=params, timeout=30)

    # Step 4. 解析响应为 Python 字典后，再序列化为 JSON 字符串返回，供后续链继续处理
    data = response.json()
    return json.dumps(data)

# 本地测试：单参数工具可直接传值；若和更通用的工具调用风格保持一致，也可传 {"loc": "..."}
# result = get_weather.invoke("shanghai")
result = get_weather.invoke("beijing")
print(result)
