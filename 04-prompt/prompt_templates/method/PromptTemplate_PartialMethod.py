"""
【案例】PromptTemplate 的 partial() 方法

对应教程章节：第 13 章 - 提示词与消息模板 → 6、文本提示词模板（PromptTemplate）

知识点速览：
- `partial(...)` 会返回一个新的 `PromptTemplate`，原模板本身不会被修改。
- 它适合把长期稳定的变量先固定下来，例如角色、人设、固定规则等，再多次填充变化部分。
- 和 `partial_variables` 的关系可以理解为：一个是在创建时固定，一个是在使用过程中继续固定。
"""

from langchain_core.prompts import PromptTemplate

# ---------- 1. 创建带两个占位符的模板 ----------
template = PromptTemplate.from_template(
    "你是一个专业的{role}工程师，我的姓名是{name},请回答我的问题给出回答，我的问题是：{question}"
)

# ---------- 2. partial(role="python开发")：固定 role，得到「新模板」----------
# 新模板只剩 {question} 需要填，适合多轮只换问题、不换角色的场景
partial1 = template.partial(role="python开发")
partial2 = partial1.partial(name='yyy')

print(partial2)
print(type(partial2))

prompt = partial2.format(question="冒泡排序怎么写？")
print(prompt)
print(type(prompt))
"""
【输出示例】
input_variables=['question'] input_types={} partial_variables={'role': 'python开发'} template='你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}'
<class 'langchain_core.prompts.prompt.PromptTemplate'>

# 你是一个专业的python开发工程师，请回答我的问题给出回答，我的问题是：冒泡排序怎么写？
# <class 'str'>
"""
