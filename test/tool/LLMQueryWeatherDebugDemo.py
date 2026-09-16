"""
【案例】链路调试：如何看到 toolcall_chain / output_chain 的中间数据

对应教程章节：第 17 章 - Tools 工具调用 → 5.4 天气助手完整链路；第 15 章 - LCEL 与链式调用 → RunnablePassthrough

知识点速览：
- RunnablePassthrough 本身“看不到”任何数据：它的唯一职责就是把输入原样输出（in == out），没有观察/打印能力。
- 想拿中间数据，真正的用法是它的类方法 `RunnablePassthrough.assign(新字段=函数)`：
  保留上游 dict 的全部字段，再把每一步的结果作为新字段挂上去，一路透传到最终输出。
  这样 invoke 一次，就能同时拿到 question / tool_args / weather_json / answer 四个阶段的完整数据。
- 只想“边跑边看”，还可以在链路里插一个只打印、不改数据的透传节点（RunnableLambda + 返回原值），
  相当于给管道开一个观察窗，链路行为完全不变。
- 完全不想改链路结构时，用 `with_listeners(on_end=...)` 在链路外面监听（只有根 run 的输入输出），
  或在 .env 打开 LangSmith（LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY），零改代码即可逐节点看输入输出。

注意：RunnablePassthrough.assign 要求输入必须是 dict（因为要往上面挂字段），所以入口统一传 {"question": "..."}。
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputKeyToolsParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from loguru import logger

from QueryWeatherTool import get_weather

load_dotenv(encoding="utf-8")

# 初始化大模型（需支持工具调用）
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 把工具声明给模型：此时不执行工具，模型判断需要时才会返回 tool_calls
llm_with_tools = llm.bind_tools([get_weather])

# 解析器：从 AIMessage.tool_calls 中取出天气工具的入参 dict
json_parser = JsonOutputKeyToolsParser(key_name=get_weather.name, first_tool_only=True)

output_prompt = PromptTemplate.from_template(
    """
    这是一段 JSON 格式的天气数据：\n{weather_json}\n 请用简洁自然的方式将其转述给用户。案例如下：
    "北京现在天气：多云，气温 28℃，体感有点闷热（约 32℃），湿度 75%，微风（东南风 2 米/秒），
    能见度很好，大约 10 公里。建议穿短袖短裤。适合做户外运动。
    """
)
output_parser = StrOutputParser()

# 输出链：天气 JSON → 提示词 → 模型 → 自然语言
output_chain = output_prompt | llm | output_parser


# =====================================================================
# 方式一（推荐）：RunnablePassthrough.assign 把中间结果一路带到最终输出
# 每一步都只做一件事：保留已有字段 + 挂一个新字段
# =====================================================================
def build_passthrough_chain():
    # “模型 → 解析器”这段子链，输入是用户问题字符串，输出是工具入参 dict
    tool_args_chain = llm_with_tools | json_parser

    step_args = RunnablePassthrough.assign(
        # 中间数据 1、2：模型返回的 tool_calls，经解析器变成工具入参
        tool_args=lambda r: tool_args_chain.invoke(r["question"])
    )
    step_tool = RunnablePassthrough.assign(
        # 中间数据 3：工具执行的原始返回（天气 JSON 字符串）
        weather_json=lambda r: get_weather.invoke(r["tool_args"])
    )
    step_answer = RunnablePassthrough.assign(
        # 中间数据 4：最终给用户的自然语言
        answer=lambda r: output_chain.invoke({"weather_json": r["weather_json"]})
    )
    return step_args | step_tool | step_answer


# =====================================================================
# 方式二：链路里插“观察窗”，只打印、不改数据（等价于带副作用的 Passthrough）
# =====================================================================
def peek(label: str) -> RunnableLambda:
    """返回一个透传节点：打印当前流经的数据，然后原样交给下一个节点。"""

    def _peek(x):
        logger.info(f"【{label} {type(x)}】{x}")
        return x  # 关键：原样返回，链路行为不变

    return RunnableLambda(_peek)


# 原始链路的“可观察版”：把 llm_with_tools 输出的 AIMessage、解析出的入参、工具结果都打印出来
toolcall_chain = (
    peek("① 用户问题")
    | llm_with_tools
    | peek("② 模型原始输出（含 tool_calls）")
    | json_parser
    | peek("③ 解析出的工具入参")
    | get_weather
    | peek("④ 工具返回的天气 JSON")
)

full_chain_with_peek = (
    toolcall_chain
    | RunnableLambda(lambda x: {"weather_json": x})
    | peek("⑤ 输出链入参")
    | output_chain
    | peek("⑥ 最终回答")
)

# 方式三：链路一行都不改，只在外面挂监听器
# 原始链路（与你现在的 LLMQueryWeatherDemo.py 完全一致，没有任何插桩）
raw_toolcall_chain = llm_with_tools | json_parser | get_weather
raw_full_chain = raw_toolcall_chain | (lambda x: {"weather_json": x}) | output_chain

# 注意：with_listeners 只在“整条链的根 run”上触发，拿到的是链路级 inputs/outputs（不含每个中间节点）
# 若要逐节点事件，用 `chain.astream_events(input, version="v2")` 异步遍历，或直接开 LangSmith
full_chain_with_listener = raw_full_chain.with_listeners(
    on_end=lambda run: logger.info(f"[listener] {run.name} 耗时={run.latency}s 输入={run.inputs} 输出={run.outputs}")
)

if __name__ == "__main__":
    question = "今天徐州的天气如何？"

    # 方式二：控制台逐步打印每个节点的中间数据
    logger.info("=== 方式二：链内插桩（peek）===")
    logger.info(full_chain_with_peek.invoke(question))

    # 方式一：一次 invoke 拿到结构化字典，四个阶段的数据全在里面
    logger.info("=== 方式一：RunnablePassthrough.assign 透传收集 ===")
    result = build_passthrough_chain().invoke({"question": question})
    for key, value in result.items():
        logger.info(f"{key} => {value}")

    # 方式三：链路结构不动，只在外面监听
    logger.info("=== 方式三：with_listeners 外部监听 ===")
    full_chain_with_listener.invoke(question)

"""
【输出示例】（以下为实际跑通的结果）
# 方式二：链内插桩，每个节点的中间数据都会打印一次
2026-09-16 10:47:53 | INFO | 【① 用户问题】今天徐州的天气如何？
2026-09-16 10:47:54 | INFO | 【② 模型原始输出（含 tool_calls）】content='' ... tool_calls=[{'name': 'get_weather', 'args': {'location': 'Xuzhou'}, 'id': 'call_7139a7a9d94d463385a456', 'type': 'tool_call'}]
2026-09-16 10:47:54 | INFO | 【③ 解析出的工具入参】{'location': 'Xuzhou'}
2026-09-16 10:47:55 | INFO | 【④ 工具返回的天气 JSON】{"coord": {"lon": 117.1571, "lat": 34.1805}, "weather": [{"id": 804, "main": "Clouds", "description": "\u9634\uff0c\u591a\u4e91"}], "main": {"temp": 27.21, "feels_like": 26.82, "humidity": 36}, ...}
2026-09-16 10:47:55 | INFO | 【⑤ 输出链入参】{'weather_json': '{"coord": ...}'}
2026-09-16 10:47:59 | INFO | 【⑥ 最终回答】徐州现在天气：阴，多云，气温 27.2℃，体感较舒适（约 26.8℃），湿度 36%……

# 方式一：一次 invoke 拿到四个阶段的全部中间数据（返回的是 dict）
question     => 今天徐州的天气如何？
tool_args    => {'location': 'Xuzhou'}
weather_json => {"coord": {"lon": 117.1571, "lat": 34.1805}, ... "cod": 200}
answer       => 徐州现在天气：阴，多云，气温 27.2℃，体感较舒适（约 26.8℃）……

# 方式三：链路不变，只在结束时拿到整条链的输入/输出与耗时
[listener] RunnableSequence 耗时=6.13s 输入={'input': '今天徐州的天气如何？'} 输出={'output': '徐州现在天气：阴，多云……'}
"""
