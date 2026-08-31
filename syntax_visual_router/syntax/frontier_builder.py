"""
句法模块: Bottom-up Frontier Builder

技术指南 §16-17:
  从 Dependency Tree 构建 bottom-up frontiers。
  同一层 (同一 depth) 的所有关系并行处理，
  全部使用相同的 old bank，避免遍历顺序偏差。

示例:
  young → man → riding
  brown → horse → riding

  Round 1: young→man, brown→horse  (depth 0→1 的边)
  Round 2: man→riding, horse→riding (depth 1→2 的边)
"""
from collections import defaultdict
from typing import List, Tuple, Dict
from .dependency_parser import DepEdge, DepNode


def compute_node_depth(
    nodes: List[DepNode],
    edges: List[DepEdge],
) -> Dict[int, int]:
    """
    计算每个节点的 depth (拓扑深度)，自底向上。
    叶节点 depth=0，父节点的 depth = max(child_depth) + 1。

    使用 child→head 方向的 BFS：
      从入度为 0 的节点（叶子）出发，
      沿 parent 边向上传播 depth。
    """
    # 邻接表: child → head（自底向上）
    parents = defaultdict(set)

    # indegree: 有多少个 child 依赖这个 head
    indegree = {}
    for n in nodes:
        indegree[n.idx] = 0
    for e in edges:
        parents[e.child.idx].add(e.head.idx)  # child→head
        indegree[e.head.idx] += 1

    # BFS 起点: 入度为 0 的节点 (叶子，没有人依赖它们)
    depth = {}
    queue = [n.idx for n in nodes if indegree[n.idx] == 0]

    if not queue:
        for n in nodes:
            depth[n.idx] = 0
            queue.append(n.idx)
    else:
        for nid in queue:
            depth[nid] = 0

    while queue:
        nid = queue.pop(0)
        for parent_id in parents.get(nid, set()):
            new_d = depth[nid] + 1
            if parent_id not in depth or new_d > depth[parent_id]:
                depth[parent_id] = new_d
                queue.append(parent_id)

    # 兜底：未被 BFS 访问到的节点 depth=0
    for n in nodes:
        if n.idx not in depth:
            depth[n.idx] = 0

    return depth


def build_frontiers(
    edges: List[DepEdge],
) -> List[List[DepEdge]]:
    """
    将依存边按 depth 分组成 frontier。
    edge 的 "深度" = child 节点的 depth。
    frontier[0] 是叶子边 (young→man, brown→horse)。
    """
    if not edges:
        return []

    # 收集所有相关节点
    all_nodes = []
    for e in edges:
        all_nodes.append(e.child)
        all_nodes.append(e.head)

    # 计算 depth
    depth_map = compute_node_depth(all_nodes, edges)

    # 按 child depth 分组
    by_depth = defaultdict(list)
    for e in edges:
        d = depth_map.get(e.child.idx, 0)
        by_depth[d].append(e)

    # 返回排序后的 frontier 列表
    max_d = max(by_depth.keys()) if by_depth else 0
    frontiers = []
    for d in range(max_d + 1):
        if d in by_depth:
            frontiers.append(by_depth[d])
    return frontiers


# ==================== 测试 ====================
if __name__ == "__main__":
    from .dependency_parser import parse_and_filter

    caption = "A young man is riding a brown horse."
    _, edges = parse_and_filter(caption)

    frontiers = build_frontiers(edges)

    for i, frontier in enumerate(frontiers):
        print(f"Round {i+1}:")
        for e in frontier:
            print(f"  {e.child_text} --{e.fine_relation}--> {e.head_text}")
    print()
    print("期望:")
    print("  Round 1: young→man, brown→horse")
    print("  Round 2: man→riding, horse→riding")
