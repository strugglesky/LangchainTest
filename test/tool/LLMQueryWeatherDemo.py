from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv(encoding="utf-8")

import os
from langchain_core.output_parsers import JsonOutputKeyToolsParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from QueryWeatherTool import get_weather


# 初始化大模型（教程 5.4：需可调用工具的大模型）
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

llm_with_tools = llm.bind_tools([get_weather])
# 解析器：从模型输出中提取“命中的天气工具参数”，得到可直接传给 get_weather 的入参
json_parser = JsonOutputKeyToolsParser(key_name=get_weather.name, first_tool_only=True)
output_prompt = PromptTemplate.from_template(
    """
    这是一段 JSON 格式的天气数据：\n{weather_json}\n 请用简洁自然的方式将其转述给用户。案例如下：
    "北京现在天气：多云，气温 28℃，体感有点闷热（约 32℃），湿度 75%，微风（东南风 2 米/秒），
    能见度很好，大约 10 公里。建议穿短袖短裤。适合做户外运动。
    """
)
output_parser = StrOutputParser()
# 拿到天气 JSON 数据
toolcall_chain = llm_with_tools | json_parser | get_weather
# 输出链：把天气 JSON 塞进提示词，由模型转成更适合用户阅读的自然语言描述
output_chain = output_prompt | llm | output_parser

full_chain = toolcall_chain | RunnableLambda(lambda x: {"weather_json": x}) | output_chain

result = full_chain.invoke("今天徐州的天气如何？")
print(result)
