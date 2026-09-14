from loguru import logger
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

load_dotenv(encoding="utf-8")

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [{"role": "user", "content": "任意生成三种动物，以及他们的 emoji 表情"}]

class Product(BaseModel):
    """产品信息：名称、类别、简介。简介长度需 ≥ 10，由下方 validator 校验。"""

    name: str = Field(description="产品名称")
    category: str = Field(description="产品类别")
    description: str = Field(description="产品简介")

    @field_validator("description")
    def validate_description(cls, value):
        """Pydantic 校验器：description 长度必须 ≥ 10，否则抛 ValueError。"""
        if len(value) < 10:
            raise ValueError("产品简介长度必须大于等于10")
        return value

# 创建 Pydantic 输出解析器：解析结果会转成 Product 实例并做校验
parser = PydanticOutputParser(pydantic_object=Product)

# 生成「格式说明」字符串，拼进 Prompt，引导模型按 Product 的字段输出 JSON
format_instructions = parser.get_format_instructions()
logger.info(f'格式说明:\n{format_instructions}')

# 在 system 里放入 {format_instructions}，human 里放 {topic}
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个AI助手，你只能输出结构化的json数据\n{format_instructions}"),
        ("human", "请你输出标题为：{topic}的新闻内容"),
    ]
)
prompt = prompt_template.invoke({"format_instructions": format_instructions, "topic": "新能源汽车"})
logger.info(f'模型Prompt:\n{prompt}')

result = model.invoke(prompt)
logger.info(f"模型原始输出:\n{result.content}")

response = parser.invoke(result)
logger.info(f"解析后的结构化结果:\n{response}")
logger.info(f"结果类型: {type(response)}")  # <class 'Product'>



