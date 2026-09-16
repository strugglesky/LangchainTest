from langchain_core.tools import tool

@tool
def add_number(a: int,b : int) -> int:
    """
    两个整数相加
    """
    return a + b

# 直接执行工具：invoke 接收参数字典，键为参数名，值为参数值（与函数签名对应）
result = add_number.invoke({"a": 12, "b": 23})
print(result)
print()

# 查看工具元信息：这些内容正是模型后续理解工具时会重点参考的部分
print(f"{add_number.name=}\n{add_number.description=}\n{add_number.args=}")
