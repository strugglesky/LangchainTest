"""
【模块】Redis-Stack 向量库去重工具（文本完全重复 + 语义高度相似）

对应教程章节：第 19 章 - RAG 检索增强生成 → 向量库维护（扩展案例）

知识点速览：
- RAG 库反复执行 add_texts / from_documents 后，很容易写入重复内容：一种是“文本一模一样”的完全重复，
  另一种是“措辞不同但语义几乎相同”的近重复（比如“我喜欢吃苹果”和“苹果是我最喜欢吃的水果”）。
- 本模块先用 FT.SEARCH 拉取索引中的全部记录 key，再 HGETALL 取回每条记录的正文和向量，
  然后在本地完成两件事：
  1) 按正文精确分组（不依赖向量，防止不同模型写入的同文本漏判）；
  2) 按余弦相似度 >= 阈值分组合并（用语义捕获近重复）。
- 分组使用“并查集”：只要两条记录满足任一重复条件就合并到同一组；每组保留一条代表记录
  （按 key 排序取第一条，ULID 是时间有序的，即保留最早写入的那条），其余全部删除。
- 删除直接对 Redis hash key 执行 DEL；Redis-Stack 的搜索索引会自动感知 hash 被删除并同步移除索引条目。
- 注意：两两比较是 O(n^2) 的，适合几百到几千条量级的教学/小库场景；
  十万级以上的大库建议改用“每条记录做一次 KNN 向量检索 + 阈值过滤”来近似找重复对。
- 安全设计：默认是 dry-run（只打印将要删除的重复组，不真正删除）；确认无误后加 --apply 才执行删除。
"""

import argparse
from typing import Any, Dict, List

import numpy as np
import redis
from redis.commands.search.query import Query as RediSearchQuery

DEFAULT_REDIS_URL = "redis://localhost:6379"
DEFAULT_INDEX_NAME = "newsgroups"
# 余弦相似度阈值：同一段文本用同一模型 embed 出来的向量相似度约等于 1.0；
# 0.98 左右可以捕获“语义几乎一样、措辞略有差异”的近重复；只想删完全重复可调到 0.9999
DEFAULT_THRESHOLD = 0.98

# 兼容不同版本写入的字段名：langchain-redis>=0.2 默认 text/embedding；旧版 langchain_community Redis 为 content/vector
CONTENT_CANDIDATES = ("text", "content", "page_content")
EMBEDDING_CANDIDATES = ("embedding", "vector")


def decode(value: Any) -> str:
    """把 redis 返回的 bytes 安全地转成字符串（非 bytes 原样返回）"""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def fetch_all_records(client: redis.Redis, index_name: str) -> List[Dict[str, Any]]:
    """扫描索引中的全部记录，返回 [{key, content, vector}, ...]

    步骤：FT.INFO 拿总数 -> FT.SEARCH "*" 拿所有文档 key -> pipeline HGETALL 批量取正文与向量
    """
    info = client.ft(index_name).info()
    total = int(info["num_docs"])
    if total == 0:
        return []

    # no_content 只取 key，避免 FT.SEARCH 返回大字段；向量本身也不走搜索结果，统一从 hash 里取
    query = RediSearchQuery("*").no_content().paging(0, total)
    search_result = client.ft(index_name).search(query)
    keys = [doc.id.decode() if isinstance(doc.id, bytes) else doc.id for doc in search_result.docs]

    # 批量 HGETALL，减少网络往返
    with client.pipeline(transaction=False) as pipe:
        for key in keys:
            pipe.hgetall(key)
        hashes = pipe.execute()

    records = []
    for key, hash_data in zip(keys, hashes):
        if not hash_data:
            continue
        fields = {decode(k): v for k, v in hash_data.items()}
        content_field = next((f for f in CONTENT_CANDIDATES if f in fields), None)
        embedding_field = next((f for f in EMBEDDING_CANDIDATES if f in fields), None)
        if content_field is None or embedding_field is None:
            # 不是本工具认识的向量记录，跳过（如 metadata-only 文档）
            continue
        # 向量在 hash 中按小端 float32 的二进制存储，直接零拷贝解析
        vector = np.frombuffer(fields[embedding_field], dtype="<f4")
        records.append({"key": key, "content": decode(fields[content_field]), "vector": vector})
    return records


class UnionFind:
    """并查集：把“互为重复”的记录合并到同一组"""

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩（隔代）
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def find_duplicate_groups(records: List[Dict[str, Any]], threshold: float) -> List[List[int]]:
    """返回重复分组（每组是记录下标列表，长度 > 1 才返回）

    重复判定规则（满足其一即合并）：
    1) 正文完全相同；2) 向量余弦相似度 >= threshold
    """
    n = len(records)
    uf = UnionFind(n)

    # 规则一：正文精确重复（不依赖向量，开销极小）
    by_content: Dict[str, int] = {}
    for i, record in enumerate(records):
        first = by_content.setdefault(record["content"], i)
        uf.union(first, i)

    # 规则二：语义近重复。向量归一化后，余弦相似度 = 内积；分块矩阵乘法避免一次性占用过多内存
    matrix = np.stack([r["vector"] for r in records]).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # 防止零向量除零
    unit = matrix / norms

    block = 256
    for start in range(0, n, block):
        end = min(start + block, n)
        sims = unit[start:end] @ unit.T  # (block, n)
        for offset in range(end - start):
            i = start + offset
            # 只比较 j > i，避免重复计算和自身比较
            js = np.where(sims[offset, i + 1:] >= threshold)[0] + i + 1
            for j in js:
                uf.union(i, int(j))

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return [sorted(members) for members in groups.values() if len(members) > 1]


def dedup(redis_url: str, index_name: str, threshold: float, apply: bool) -> None:
    client = redis.Redis.from_url(redis_url, decode_responses=False)

    try:
        records = fetch_all_records(client, index_name)
    except redis.exceptions.ResponseError as e:
        print(f"索引 [{index_name}] 不存在或不可搜索，请确认 redis-stack 已启动且索引名正确：{e}")
        return

    print(f"索引 [{index_name}] 共扫描到 {len(records)} 条向量记录，相似度阈值 = {threshold}")
    if len(records) < 2:
        print("记录数不足 2 条，无需去重。")
        return

    groups = find_duplicate_groups(records, threshold)
    if not groups:
        print("未发现重复记录，向量库已经是去重状态。")
        return

    keys_to_delete: List[str] = []
    print(f"\n共发现 {len(groups)} 组重复：")
    for gid, members in enumerate(groups, 1):
        # ULID key 时间有序，保留最早写入的一条作为代表，删除其余
        sorted_members = sorted((records[i]["key"] for i in members))
        keep_key = sorted_members[0]
        drop_keys = sorted_members[1:]
        keys_to_delete.extend(drop_keys)
        print(f"\n组 {gid}（共 {len(members)} 条，保留 {keep_key}）")
        for i in members:
            mark = "保留" if records[i]["key"] == keep_key else "删除"
            print(f"  [{mark}] {records[i]['key']} -> {records[i]['content'][:50]}")

    if not apply:
        print(f"\n当前为 dry-run 模式，未实际删除。确认无误后追加 --apply 执行删除（共 {len(keys_to_delete)} 条）。")
        return

    deleted = client.delete(*keys_to_delete)
    print(f"\n去重完成：已删除 {deleted} 条重复记录，索引剩余 {int(client.ft(index_name).info()['num_docs'])} 条。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redis-Stack 向量库去重工具（默认 dry-run）")
    parser.add_argument("--redis-url", default=DEFAULT_REDIS_URL, help="Redis 连接地址")
    parser.add_argument("--index", default=DEFAULT_INDEX_NAME, help="要去重的向量索引名")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="余弦相似度阈值（0~1）")
    parser.add_argument("--apply", action="store_true", help="真正执行删除（不加则只预览）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dedup(args.redis_url, args.index, args.threshold, args.apply)
