import os
import json
import httpx
from pathlib import Path
from typing_extensions import (
    TypedDict,
)  # Python < 3.12 下 Pydantic 要求用 typing_extensions.TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# .env 在项目根目录，从任意子目录运行脚本时都从根目录加载
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

@tool
def get_weather(loc: str) -> str:
    """
    查询即时天气函数
    :param loc: 城市英文名，如 Beijing、Shanghai。
    :return: OpenWeather API 返回的天气信息（JSON 字符串）。
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": loc,
        "appid": os.getenv("OPENWEATHER_API_KEY"),
        "units": "metric",
        "lang": "zh_cn",
    }
    response = httpx.get(url, params=params, timeout=30)
    data = response.json()
    return json.dumps(data, ensure_ascii=False)

# 定义结构化输出：Agent 最终回答会按此结构填充，便于代码中直接取字段
class WeatherCompareOutput(TypedDict):
    beijing_temp: float
    shanghai_temp: float
    hotter_city: str
    summary: str

model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# V1.0 一步创建 Agent：模型、工具、系统提示、输出格式一次传入
# 如果后面还要扩展短期记忆或拦截控制，通常会继续给 create_agent 传 checkpointer / middleware
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt=(
        "你是天气助手。"
        "当用户询问多个城市天气时，"
        "你需要分别调用工具获取数据，并进行比较分析。"
    ),
    response_format=WeatherCompareOutput,
)

# 调用 Agent，返回结果中包含 messages 与 structured_response（若指定了 response_format）
# 这里先用 invoke 看最终结果；如需观察中间步骤，可在工程里改为 stream()
result = agent.invoke({"input": "请问今天北京和上海的天气怎么样，哪个城市更热?"})
print(result)
print()
print(json.dumps(result["structured_response"], ensure_ascii=False, indent=2))