from langchain_core.tools import tool
from loguru import logger
from pydantic import BaseModel, Field
from pydantic.v1 import StrictInt


# Pydantic 模型：定义“工具参数接口”，字段 description 会进入工具参数 schema
class FieldInfo(BaseModel):
    """
    定义加法运算所需的参数结构
    """
    a: int = Field(description="被加数")
    b: int = Field(description="加数")

@tool(args_schema=FieldInfo)
def add_number(a: int, b: int) -> int:
    """
    两个整数相加
    """
    return a + b

# 打印工具属性：带 args_schema 时，args 中会包含 Field 的 description
logger.info(f"name = {add_number.name}")
logger.info(f"args = {add_number.args}")
logger.info(f"description = {add_number.description}")
logger.info(f"return_direct = {add_number.return_direct}")

# 调用工具：传入字典，Pydantic 会做类型校验与转换
res = add_number.invoke({"a": 1, "b": 2})
logger.info(res)