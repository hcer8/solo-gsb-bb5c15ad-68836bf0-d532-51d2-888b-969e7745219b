"""冲突图脆弱性分析：Tarjan low-link 一次遍历求全部割点与桥。

冲突图
------
顶点为批次内全部引物（顶点号即批次位置，0 起），边为命中所选风险等级的
**异二聚体**配对；自二聚体不参与连通性（由调用方另行汇总为固有自身风险）。
平行重复边与自环在进入邻接结构前去重/忽略。图可不连通：孤立顶点视为大小
为 1 的连通分量。

Tarjan low-link
---------------
对每个连通分量做一次深度优先搜索（起点按位置升序、邻居按位置升序访问，
遍历顺序完全确定），维护发现时间 ``disc`` 与 low-link 值 ``low``：

* 非根顶点 u 是**割点** ⟺ 存在 DFS 子节点 v 使 ``low[v] >= disc[u]``。
  此时每个满足条件的 v 的子树在移除 u 后各自成为一个分裂组；u 所在连通
  分量的其余顶点（祖先侧与未分离子树，非空）构成最后一个分裂组；
* 根顶点 r 是**割点** ⟺ DFS 子节点数 ≥ 2，每个子节点的子树即一个分裂组；
* 边 (u, v)（v 为 u 的 DFS 子节点）是**桥** ⟺ ``low[v] > disc[u]``。
  桥的两侧即 v 的子树与所在连通分量的其余顶点。

DFS 过程中按完成顺序聚合子树成员，同一遍遍历即得到全部分裂组与桥两侧，
无需二次搜索。输出约定：割点按位置升序；分裂组内成员按位置升序、组间按
组内最小位置升序；桥按两端位置 (a, b) 升序（a < b），``side_a`` 为含 a
的一侧。全部比较均为全序，相同输入必得相同结果。
"""

from __future__ import annotations

from typing import NamedTuple


class ArticulationPoint(NamedTuple):
    """一个割点及其移除后所在连通分量分裂出的各组（位置升序）。"""

    vertex: int
    split_groups: tuple[tuple[int, ...], ...]


class Bridge(NamedTuple):
    """一座桥：a < b 为两端位置，side_a / side_b 为两侧成员（位置升序）。"""

    a: int
    b: int
    side_a: tuple[int, ...]
    side_b: tuple[int, ...]


class FragilityMap(NamedTuple):
    """全图脆弱性分析结果（割点、桥均按批次位置确定性排序）。"""

    component_count: int
    articulation_points: tuple[ArticulationPoint, ...]
    bridges: tuple[Bridge, ...]


def find_fragility(n: int, edges: list[tuple[int, int]]) -> FragilityMap:
    """在 n 个顶点的无向冲突图上求全部割点与桥。

    ``edges`` 元素为 ``(u, v)``，端点顺序任意；自环与平行重复边自动忽略。
    顶点编号为 0..n-1；没有任何边的顶点计入连通分量数（大小为 1）。
    """
    adjacency: list[list[int]] = [[] for _ in range(n)]
    seen: set[tuple[int, int]] = set()
    for raw_u, raw_v in edges:
        u, v = (raw_u, raw_v) if raw_u < raw_v else (raw_v, raw_u)
        if u == v or (u, v) in seen:
            continue  # 自环与平行重复边不参与连通性
        seen.add((u, v))
        adjacency[u].append(v)
        adjacency[v].append(u)
    for neighbors in adjacency:
        neighbors.sort()

    disc = [0] * n
    low = [0] * n
    parent = [-1] * n
    children: list[list[int]] = [[] for _ in range(n)]
    finish_order: list[int] = []
    roots: list[int] = []
    timer = 0

    for start in range(n):
        if disc[start]:
            continue
        roots.append(start)
        timer += 1
        disc[start] = low[start] = timer
        # 显式栈迭代 DFS（避免深图触发递归上限）；栈项为 [顶点, 下一邻居下标]
        stack: list[list[int]] = [[start, 0]]
        while stack:
            top = stack[-1]
            u, idx = top[0], top[1]
            if idx < len(adjacency[u]):
                v = adjacency[u][idx]
                top[1] += 1
                if disc[v] == 0:
                    parent[v] = u
                    children[u].append(v)
                    timer += 1
                    disc[v] = low[v] = timer
                    stack.append([v, 0])
                elif v != parent[u] and disc[v] < low[u]:
                    # 回边（无向图 DFS 不存在横叉边）；父边只跳过一次
                    low[u] = disc[v]
            else:
                stack.pop()
                finish_order.append(u)
                p = parent[u]
                if p != -1 and low[u] < low[p]:
                    low[p] = low[u]

    # 子树成员聚合：finish_order 为先子后父的后序，子树成员并入父节点
    members: list[list[int]] = [[v] for v in range(n)]
    for v in finish_order:
        p = parent[v]
        if p != -1:
            members[p].extend(members[v])

    articulation_points: list[ArticulationPoint] = []
    bridges: list[Bridge] = []

    for root in roots:
        component = sorted(members[root])
        component_set = set(component)
        for u in component:
            for v in children[u]:
                if low[v] > disc[u]:
                    # 桥 (u, v)：一侧为 v 的子树，另一侧为分量其余顶点
                    subtree = set(members[v])
                    subtree_sorted = tuple(sorted(subtree))
                    rest_sorted = tuple(sorted(component_set - subtree))
                    a, b = (u, v) if u < v else (v, u)
                    if a in subtree:
                        side_a, side_b = subtree_sorted, rest_sorted
                    else:
                        side_a, side_b = rest_sorted, subtree_sorted
                    bridges.append(Bridge(a, b, side_a, side_b))

            if parent[u] == -1:
                # 根：每个 DFS 子节点的子树即一个分裂组
                if len(children[u]) >= 2:
                    groups = sorted(
                        tuple(sorted(members[v])) for v in children[u]
                    )
                    articulation_points.append(
                        ArticulationPoint(u, tuple(groups))
                    )
            else:
                separated = [v for v in children[u] if low[v] >= disc[u]]
                if separated:
                    groups = [tuple(sorted(members[v])) for v in separated]
                    detached = set().union(
                        *(set(members[v]) for v in separated)
                    )
                    # 其余顶点（含祖先侧，非空）为最后一个分裂组
                    groups.append(tuple(sorted(component_set - detached - {u})))
                    groups.sort()
                    articulation_points.append(
                        ArticulationPoint(u, tuple(groups))
                    )

    articulation_points.sort(key=lambda ap: ap.vertex)
    bridges.sort(key=lambda br: (br.a, br.b))
    return FragilityMap(
        component_count=len(roots),
        articulation_points=tuple(articulation_points),
        bridges=tuple(bridges),
    )
