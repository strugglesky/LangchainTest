import json
import os
import time
import httpx
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from loguru import logger
load_dotenv()

from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# OpenWeather 必须经由代理访问（本机系统代理为 127.0.0.1:7897，httpx 默认 trust_env=True 会自动沿用），
# 代理节点偶发断链会抛 ConnectError/SSL UNEXPECTED_EOF，因此这里做超时 + 重试
WEATHER_TIMEOUT = 10
WEATHER_RETRIES = 3

@tool
def get_weather(loc):
    """
    查询即时天气函数

    :param loc: 必要参数，字符串类型，表示查询天气的城市名称；中国城市需用英文名，如 Beijing、Shanghai。
    :return: OpenWeather API 返回的天气信息，JSON 序列化后的字符串；请求失败时返回带 error 字段的 JSON 字符串。
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": loc,
        "appid": os.getenv("OPENWEATHER_API_KEY"),
        "units": "metric",
        "lang": "zh_cn",
    }
    # 如需绕开系统代理或指定节点，在 .env 里配 WEATHER_PROXY=http://127.0.0.1:7897 即可
    request_kwargs = {"timeout": WEATHER_TIMEOUT}
    if os.getenv("WEATHER_PROXY"):
        request_kwargs["proxy"] = os.getenv("WEATHER_PROXY")

    last_error = None
    for attempt in range(1, WEATHER_RETRIES + 1):
        try:
            response = httpx.get(url, params=params, **request_kwargs)
            response.raise_for_status()
            data = response.json()
            # print(json.dumps(data))
            return json.dumps(data)
        except Exception as e:  # 代理抖动、超时、HTTP 错误统一按失败处理
            last_error = e
            logger.warning(
                f"[get_weather({loc})] 第 {attempt}/{WEATHER_RETRIES} 次请求失败："
                f"{type(e).__name__}: {str(e)[:100]}"
            )
            time.sleep(attempt)  # 逐次拉长退避，给代理节点恢复时间

    # 关键：不抛异常，而是把错误当成工具返回值，Agent 循环不会中断，
    # 这条错误也会作为 observation 写进 agent_scratchpad，便于观察与自我修正
    return json.dumps(
        {
            "error": f"天气接口请求失败：{type(last_error).__name__}",
            "detail": str(last_error)[:200],
            "city": loc,
        },
        ensure_ascii=False,
    )

def peek(label: str) -> RunnableLambda:
    """返回一个透传节点：打印当前流经的数据，然后原样交给下一个节点。"""

    def _peek(x):
        logger.info(f"【{label} {type(x)}】{x}")
        return x  # 关键：原样返回，链路行为不变

    return RunnableLambda(_peek)

# 初始化大模型，用于理解用户问题并决定是否调用工具、如何组合结果
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义 Agent 的对话结构：system 定角色，human 为用户输入，
# placeholder 供 Executor 填入中间推理与工具调用记录
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是天气助手，请根据用户的问题，给出相应的天气信息"),
        ("human", "{input}"),
        (
            "placeholder",
            "{agent_scratchpad}",
        ),  # V0.3 必备：Agent 的「草稿本」，记录多轮推理与工具输出
    ]
)

tools = [get_weather]
agent = create_tool_calling_agent(llm, tools, prompt)
# return_intermediate_steps 在 langchain-classic 1.x 中默认为 False；
# 打开后若改用 invoke()，可直接从 result["intermediate_steps"] 拿到草稿本原料
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
)

question = "请问今天北京和上海的天气怎么样，哪个城市更热？"

# 观察点一：运行中实时查看。Executor 每完成一轮工具调用，
# 就会把 (AgentAction, 工具返回) 追加到 agent_scratchpad，stream() 会逐步把这些增量吐出来
logger.info("=" * 20 + " 运行中观察 agent_scratchpad " + "=" * 20)
intermediate_steps = []  # 自己累积，相当于每轮送入草稿本的内容
seen = set()  # 同一轮并行发起多个工具调用时，同一条 AIMessage 会被多个 chunk 重复带走
for chunk in agent_executor.stream({"input": question}):
    if "output" in chunk:  # 最后一块：最终回答（它自带的 messages 是空列表，需先判）
        logger.info(f"【最终回答】{str(chunk['output'])[:60]} ...")
        continue
    for msg in chunk.get("messages", []):  # AIMessage(带 tool_calls) 或 ToolMessage，即草稿本的新增行
        msg_key = (
            msg.type,
            str(msg.content),
            tuple(tc["id"] for tc in getattr(msg, "tool_calls", [])),
        )
        if msg_key in seen:
            continue
        seen.add(msg_key)
        logger.info(
            f"【草稿本新增】type={msg.type} "
            f"tool_calls={getattr(msg, 'tool_calls', [])} "
            f"content={str(msg.content)[:60]}"
        )
    # 注意：stream() 内部以 yield_actions=True 运行，只吐 actions / steps / 最终 output，
    # 不会吐 intermediate_step；AgentStep 同时带着 action 与 observation，正好是草稿本的一行
    if "steps" in chunk:
        intermediate_steps.extend(
            (step.action, step.observation) for step in chunk["steps"]
        )
        logger.info(f"【草稿本已有 {len(intermediate_steps)} 步】")

# 观察点二：回放最后一步时 {agent_scratchpad} 被填充的真实内容
# （与上面 stream 的 chunk.messages 略有差异：tools formatter 生成的是带 tool_call_id 的 ToolMessage）
logger.info("=" * 20 + " 回放送入模型的消息 " + "=" * 20)
for msg in prompt.format_messages(
    input=question,
    agent_scratchpad=format_to_tool_messages(intermediate_steps),
):
    # 发起工具调用那条 AIMessage 的 content 为空，信息全在 tool_calls 里，一并打出来
    calls = [
        f"{tc['name']}({tc['args']})" for tc in getattr(msg, "tool_calls", [])
    ]
    logger.info(f"【渲染后】{msg.type}: {str(msg.content)[:80]} {calls}")






