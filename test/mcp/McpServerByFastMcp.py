# pip install mcp
# pip install pywin32  # Windows 下部分功能需要
from loguru import logger
from mcp.server.fastmcp import FastMCP

# 创建 MCP 实例，对应「MCP 服务器」角色
mcp = FastMCP("McpDemo01")


# 为 MCP 实例添加工具：最典型的“可执行动作”
@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b


# 为 MCP 实例添加资源：资源更像“可读取内容”，常由宿主决定是否拿来做上下文
@mcp.resource("greeting://default")
def get_greeting() -> str:
    return "Hello from static resource!"


# 为 MCP 实例添加提示词模板：更像“可复用的提示词入口”或工作流模板
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    styles = {
        "friendly": "写一句友善的问候",
        "formal": "写一句正式的问候",
        "casual": "写一句轻松的问候",
    }
    return f"为{name}{styles.get(style, styles['friendly'])}"


if __name__ == "__main__":
    # STDIO 模式：与主进程通过标准输入/输出通信，适合本地集成。
    # 注意：直接运行本脚本时，没有 MCP 客户端连接，stdin 收到终端输入（如回车）会被当 JSON 解析，
    # 导致 Invalid JSON / Internal Server Error，属预期现象。正确用法是由 Cursor/Claude 等 MCP 客户端
    # 启动本进程并接管 stdin/stdout；如果想做网络化示例，可看仓库里的 McpServerWeatherByFastMCP.py。
    logger.info(f'{mcp.list_resources()}')
    mcp.run(transport="stdio")