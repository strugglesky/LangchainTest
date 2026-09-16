from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
import os
import redis
from loguru import logger

try:
    from langchain_redis import RedisChatMessageHistory

    USE_LANGCHAIN_REDIS = True
except ModuleNotFoundError:
    from langchain_community.chat_message_histories import RedisChatMessageHistory

    USE_LANGCHAIN_REDIS = False

# 支持环境变量 REDIS_URL；未设置时默认 localhost:6379（标准 Redis），教程 Docker 可能用 26379
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
FORCE_SAVE = os.getenv("REDIS_FORCE_SAVE", "0") == "1"


def _check_redis():
    """启动时检查 Redis 是否可达，不可达时给出明确提示后退出。"""
    try:
        r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        r.close()
    except (redis.ConnectionError, redis.ResponseError) as e:
        logger.error(
            "Redis 连接失败（{}）。请先启动 Redis，例如：\n"
            "  docker run -d -p 6379:6379 redis\n"
            "若使用其他端口，可设置环境变量：REDIS_URL=redis://localhost:端口",
            REDIS_URL,
        )
        raise SystemExit(1) from e

_check_redis()

# 原生 Redis 客户端，decode_responses=True 使返回值为 str 而非 bytes
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
logger.info(
    "Redis 历史实现：{} | REDIS_URL={}",
    "langchain-redis" if USE_LANGCHAIN_REDIS else "langchain-community（兼容回退）",
    REDIS_URL,
)

llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder("history"), ("human", "{question}")]
)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """为每个 session_id 创建/返回对应的 Redis 历史实例，实现持久化存储。"""
    if USE_LANGCHAIN_REDIS:
        return RedisChatMessageHistory(
            session_id=session_id,
            redis_url=REDIS_URL,
        )
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
    )

chain = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
config = RunnableConfig(configurable={"session_id": "user-001"})
print("开始对话（输入 'quit' 退出）")
while True:
    question = input("\n输入问题：")
    if question.lower() in ["quit", "exit", "q"]:
        break
    response = chain.invoke({"question": question}, config)
    logger.info(f"AI回答:{response.content}")
    # 可选：把 Redis 当前内存快照刷到磁盘，方便演示“Redis 重启后仍能恢复”。
    # 这不是多轮记忆生效的必要条件，真实项目也不建议在每轮对话后都手动 SAVE。
    if FORCE_SAVE:
        redis_client.save()


