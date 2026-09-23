import json
from loguru import logger

# 「连接」方式：通过导入获取服务端的 mcp 对象，直接读取其工具注册表，并非真实网络连接
from McpServer import mcp

class MCPWeatherClient:
    """教学版客户端：直接访问服务端注册表，用来观察最小调用链路。"""

    def __init__(self, mcp_instance):
        self.mcp_instance = mcp_instance
        # 获取服务端已注册的所有工具（字典：工具名 -> 可调用函数）
        # 真实 MCP 客户端不会直接碰 _tools，而是先握手/发现能力，再通过协议发起调用
        self.available_tools = mcp_instance._tools

    def check_tool_availability(self, tool_name: str) -> bool:
        """检查指定工具是否在服务端已注册，避免调用不存在的工具"""
        is_available = tool_name in self.available_tools
        if is_available:
            logger.info(f"工具 '{tool_name}' 可用")
        else:
            logger.warning(f"工具 '{tool_name}' 未在服务端注册")
        return is_available

    def call_get_weather(self, city: str) -> str or None:
        """调用服务端的 get_weather 工具，查询指定城市天气"""
        tool_name = "get_weather"
        if not self.check_tool_availability(tool_name):
            return None

        try:
            # 直接调用服务端已注册的工具函数。
            # 真实项目里，这一步通常由 MCP 客户端经由 stdio 或 HTTP 传输层去完成。
            weather_result = self.available_tools[tool_name](city)
            logger.info(
                f"成功获取 {city} 天气数据，返回结果长度：{len(weather_result)}"
            )
            return weather_result
        except Exception as exc:
            logger.error(f"调用 {tool_name} 工具失败：{str(exc)}")
            return None


def run_client_demo():
    """客户端演示：初始化客户端，依次查询多城市天气并格式化输出"""
    logger.info("初始化 MCP 天气客户端...")
    client = MCPWeatherClient(mcp)

    # 调用天气查询工具（支持 Beijing、Shanghai、Guangzhou 等英文城市名）
    target_cities = ["Xuzhou", "Cangzhou"]
    for city in target_cities:
        logger.info(f"\n========== 查询 {city} 天气 ==========")
        weather_data = client.call_get_weather(city)
        if weather_data:
            # 格式化输出结果（可选，方便阅读）
            formatted_data = json.dumps(
                json.loads(weather_data), indent=4, ensure_ascii=False
            )
            print(f"格式化天气结果：\n{formatted_data}")
        print("-" * 50)

if __name__ == "__main__":
    logger.info("启动 MCP 天气客户端...")
    run_client_demo()



