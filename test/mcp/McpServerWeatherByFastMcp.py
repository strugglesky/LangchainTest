import json
import os
import time

# pip install mcp httpx python-dotenv
from dotenv import load_dotenv
from loguru import logger
from mcp.server.fastmcp import FastMCP
import httpx
import anyio

load_dotenv()

# run() 只接受 transport / mount_path；host、port 等网络绑定信息在构造函数里指定
mcp = FastMCP(
    "WeatherServerSSE",  # "WeatherServerSSE" 就是你自己起的名，可改成 "MyWeather" 等
    host="127.0.0.1",    # 监听地址，写在构造函数中
    port=8000,           # 监听端口，写在构造函数中
)

# 部分网络环境的代理/网关会在 TLS 握手阶段间歇性干扰证书校验（报 SSL: UNEXPECTED_EOF_WHILE_READING），
# 策略：首次正常校验失败后，降级为跳过证书校验并最多重试 3 次（每次间隔 1 秒）
def get_with_ssl_fallback(url: str, params: dict, timeout: int = 10, retries: int = 3) -> httpx.Response:
    last_exc = None
    for attempt in range(1, retries + 1):
        verify = attempt == 1  # 第一次优先完整 HTTPS 校验，后续降级
        try:
            return httpx.get(url, params=params, timeout=timeout, verify=verify)
        except httpx.ConnectError as exc:
            last_exc = exc
            if "SSL" not in str(exc):
                raise
            logger.warning(f"第 {attempt} 次请求 SSL 受阻{'，降级为 verify=False ' if verify else ' '}(第 {attempt}/{retries} 次)：{exc}")
            time.sleep(1)
    raise last_exc

@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的即时天气信息。city 为城市英文名，如 Beijing、Shanghai。"""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": os.getenv("OPENWEATHER_API_KEY"),
        "units": "metric",
        "lang": "zh_cn",
    }
    resp = get_with_ssl_fallback(url, params, timeout=10)
    data = resp.json()
    return json.dumps(data, ensure_ascii=False)

# list_tools() 是 async 方法，await 后得到一个 list[MCPTool]，用普通 for 遍历
async def print_tools():
    tools = await mcp.list_tools()
    for tool in tools:
        logger.info(f"工具名：{tool.name}，描述：{tool.description}")

if __name__ == "__main__":
    # run() 只传 transport（可选 mount_path），host/port 已在构造函数中配置
    # 这里启动后，mcp.json 中的 weather 服务就可以按约定地址连到它。
    anyio.run(print_tools)  # 先在同一进程里跑一次事件循环，打印工具列表
    mcp.run(transport="sse")
