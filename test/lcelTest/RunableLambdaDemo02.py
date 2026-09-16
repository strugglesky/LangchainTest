from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough()
res1 = chain.invoke({"topic": "movies"})
# 输出: {'topic': 'movies'}
print(res1)

# 输入包含 text 字段，assign 会添加一个 upper_text 字段
runnable = RunnablePassthrough.assign(
    upper_text=lambda x: get_upperText(x)
)
def get_upperText(x):
    return "1234"
res2 = runnable.invoke({"text": "hello langchain"})
print(res2)
# 输出: {'text': 'hello langchain', 'upper_text': 'HELLO LANGCHAIN'}