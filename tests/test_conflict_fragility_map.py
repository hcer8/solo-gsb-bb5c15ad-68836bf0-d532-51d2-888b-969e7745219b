"""conflict-fragility-map 接口端到端测试与 Tarjan 割点/桥单测。

测试批次（ct=4、tt=3，风险结构已用 analyze_pair 独立核算）：

* 星型分量 {0,1,2,3}：S1/S2/S3 与 CEN 之间各一条 high 边，
  CEN(3) 是唯一割点，三条星边均为桥；
* 环形分量 {4,5,6,7}：G1/G2 与 C1/C2 构成 K2,2（4 条 high 边），无割点无桥；
* H(8)：自二聚体 high（intrinsic_self_risks），图中为孤立顶点；
* ISO(9)：完全孤立顶点；
* M1(10)-M2(11)：唯一一条 medium 边（两点分量，边为桥、端点非割点）。
"""

from __future__ import annotations

import os
import random

import pytest

from app.fragility import find_fragility


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("fragility_db")
    os.environ["PRIMER_DB_PATH"] = str(tmp_dir / "test.db")

    from app import db, main

    db.init_db()
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


PAYLOAD = {
    "primers": [
        {"name": "S1", "sequence": "ACACAAAAAA"},
        {"name": "S2", "sequence": "ACACAAAAAA"},
        {"name": "S3", "sequence": "ACACAAAAAA"},
        {"name": "CEN", "sequence": "TTTTTTTTTT"},
        {"name": "G1", "sequence": "ACACAGGGGG"},
        {"name": "G2", "sequence": "ACACAGGGGG"},
        {"name": "C1", "sequence": "ACACACCCCC"},
        {"name": "C2", "sequence": "ACACACCCCC"},
        {"name": "H", "sequence": "ACGTACGTAC"},
        {"name": "ISO", "sequence": "ACACACACAC"},
        {"name": "M1", "sequence": "AAGCCGAAGA"},
        {"name": "M2", "sequence": "AACGGCAAGA"},
    ],
    "continuous_threshold": 4,
    "three_prime_threshold": 3,
}


@pytest.fixture(scope="module")
def batch_id(client):
    return client.post("/api/batches", json=PAYLOAD).json()["id"]


def _map(client, batch_id, risk_levels):
    return client.post(
        f"/api/batches/{batch_id}/conflict-fragility-map",
        json={"risk_levels": risk_levels},
    )


# ---------- high 等级：星型割点与桥 ----------

def test_high_level_structure(client, batch_id):
    resp = _map(client, batch_id, ["high"])
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["batch_id"] == batch_id
    assert data["risk_levels"] == ["high"]
    assert data["primer_count"] == 12
    # 3 条星边 + 4 条环边；M1-M2 为 medium 不计入
    assert data["conflict_edge_count"] == 7
    # 星型、环形、{H}、{ISO}、{M1}、{M2} 共 6 个连通分量
    assert data["component_count"] == 6
    assert data["articulation_point_count"] == 1
    assert data["bridge_count"] == 3

    (ap,) = data["articulation_points"]
    assert ap["position"] == 3
    assert ap["name"] == "CEN"
    # 移除 CEN 后星型分量分裂为 {S1} {S2} {S3}，全图分量数 6-1+3=8
    assert ap["split_group_count"] == 3
    assert ap["component_count_after_removal"] == 8
    groups = [
        [(m["position"], m["name"]) for m in g["members"]]
        for g in ap["split_groups"]
    ]
    assert groups == [
        [(0, "S1")],
        [(1, "S2")],
        [(2, "S3")],
    ]

    bridges = data["bridges"]
    assert [(b["position_a"], b["position_b"]) for b in bridges] == [
        (0, 3),
        (1, 3),
        (2, 3),
    ]
    for b in bridges:
        assert b["risk_level"] == "high"
        assert b["name_b"] == "CEN"
        leaf = b["position_a"]
        assert [m["position"] for m in b["side_a"]] == [leaf]
        assert [m["position"] for m in b["side_b"]] == [
            p for p in (0, 1, 2, 3) if p != leaf
        ]

    # H 自二聚体 high：不参与连通性，单独返回
    assert data["intrinsic_self_risks"] == [
        {"position": 8, "name": "H", "risk_level": "high"}
    ]


# ---------- high+medium：medium 边成为两点分量的桥 ----------

def test_high_medium_includes_medium_bridge(client, batch_id):
    resp = _map(client, batch_id, ["high", "medium"])
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["conflict_edge_count"] == 8
    # 星型、环形、{M1,M2}、{H}、{ISO} 共 5 个连通分量
    assert data["component_count"] == 5
    assert data["articulation_point_count"] == 1
    ap = data["articulation_points"][0]
    assert ap["position"] == 3
    assert ap["component_count_after_removal"] == 7

    assert data["bridge_count"] == 4
    medium_bridge = data["bridges"][3]
    assert (medium_bridge["position_a"], medium_bridge["position_b"]) == (10, 11)
    assert (medium_bridge["name_a"], medium_bridge["name_b"]) == ("M1", "M2")
    assert medium_bridge["risk_level"] == "medium"
    assert [m["name"] for m in medium_bridge["side_a"]] == ["M1"]
    assert [m["name"] for m in medium_bridge["side_b"]] == ["M2"]

    # 两点分量的端点不是割点（移除一端后另一端自成分量，分量数不变）
    assert {ap["position"] for ap in data["articulation_points"]} == {3}


# ---------- 仅 medium / 仅 low 的过滤效果 ----------

def test_medium_only(client, batch_id):
    data = _map(client, batch_id, ["medium"]).json()
    assert data["conflict_edge_count"] == 1
    # {M1,M2} 之外 10 个孤立顶点
    assert data["component_count"] == 11
    assert data["articulation_points"] == []
    assert [ (b["position_a"], b["position_b"]) for b in data["bridges"] ] == [
        (10, 11)
    ]
    # 没有 medium 自二聚体
    assert data["intrinsic_self_risks"] == []


def test_low_only_dense_graph_has_no_fragility(client, batch_id):
    data = _map(client, batch_id, ["low"]).json()
    # 66 个异二聚体中 8 个非 low；low 边构成 K12 去掉 8 条边的稠密图
    assert data["conflict_edge_count"] == 58
    assert data["component_count"] == 1
    assert data["articulation_points"] == []
    assert data["bridges"] == []
    # 除 H 外 11 条引物的自二聚体均为 low
    assert len(data["intrinsic_self_risks"]) == 11
    assert all(s["risk_level"] == "low" for s in data["intrinsic_self_risks"])
    assert [s["position"] for s in data["intrinsic_self_risks"]] == [
        p for p in range(12) if p != 8
    ]


def test_all_levels_complete_graph(client, batch_id):
    data = _map(client, batch_id, ["high", "medium", "low"]).json()
    assert data["conflict_edge_count"] == 66
    assert data["component_count"] == 1
    assert data["articulation_points"] == []
    assert data["bridges"] == []
    assert len(data["intrinsic_self_risks"]) == 12


# ---------- 规范化、确定性、只读性 ----------

def test_risk_levels_normalized_and_deduped(client, batch_id):
    resp = _map(client, batch_id, ["HIGH", " high ", "medium", "MEDIUM"])
    assert resp.status_code == 200, resp.text
    assert resp.json()["risk_levels"] == ["high", "medium"]


def test_deterministic_across_requests(client, batch_id):
    r1 = _map(client, batch_id, ["high", "medium"]).json()
    r2 = _map(client, batch_id, ["high", "medium"]).json()
    assert r1 == r2
    # 风险等级顺序不影响图分析结果（risk_levels 为请求回显，除外）
    r3 = _map(client, batch_id, ["medium", "high"]).json()
    assert {k: v for k, v in r1.items() if k != "risk_levels"} == {
        k: v for k, v in r3.items() if k != "risk_levels"
    }


def test_batch_not_modified(client, batch_id):
    before = client.get(f"/api/batches/{batch_id}").json()
    resp = _map(client, batch_id, ["high", "medium", "low"])
    assert resp.status_code == 200
    after = client.get(f"/api/batches/{batch_id}").json()
    assert after == before


# ---------- 错误处理 ----------

def test_batch_not_found_404(client):
    resp = _map(client, "no-such-batch", ["high"])
    assert resp.status_code == 404


def test_invalid_risk_level_422(client, batch_id):
    resp = _map(client, batch_id, ["critical"])
    assert resp.status_code == 422


def test_empty_risk_levels_422(client, batch_id):
    resp = _map(client, batch_id, [])
    assert resp.status_code == 422


# ---------- find_fragility 单测 ----------

def _components_after_removal(n, edges, remove_v=None, remove_e=None):
    """暴力参照：删除指定顶点或边后的连通分量（BFS）。"""
    adj = [set() for _ in range(n)]
    for a, b in edges:
        if a == b:
            continue
        e = (min(a, b), max(a, b))
        if remove_e is not None and e == remove_e:
            continue
        adj[a].add(b)
        adj[b].add(a)
    seen = [False] * n
    groups = []
    for s in range(n):
        if s == remove_v or seen[s]:
            continue
        stack = [s]
        seen[s] = True
        group = []
        while stack:
            u = stack.pop()
            group.append(u)
            for v in adj[u]:
                if v == remove_v or seen[v]:
                    continue
                seen[v] = True
                stack.append(v)
        groups.append(tuple(sorted(group)))
    return groups


def test_path_graph():
    # 0-1-2-3：1、2 为割点，三条边均为桥
    fm = find_fragility(4, [(0, 1), (1, 2), (2, 3)])
    assert fm.component_count == 1
    aps = {ap.vertex: ap.split_groups for ap in fm.articulation_points}
    assert aps == {1: ((0,), (2, 3)), 2: ((0, 1), (3,))}
    assert [(b.a, b.b) for b in fm.bridges] == [(0, 1), (1, 2), (2, 3)]
    sides = {(b.a, b.b): (b.side_a, b.side_b) for b in fm.bridges}
    assert sides[(0, 1)] == ((0,), (1, 2, 3))
    assert sides[(1, 2)] == ((0, 1), (2, 3))
    assert sides[(2, 3)] == ((0, 1, 2), (3,))


def test_cycle_graph_has_no_fragility():
    fm = find_fragility(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert fm.component_count == 1
    assert fm.articulation_points == ()
    assert fm.bridges == ()


def test_star_center_is_articulation_point():
    fm = find_fragility(4, [(3, 0), (3, 1), (3, 2)])
    (ap,) = fm.articulation_points
    assert ap.vertex == 3
    assert ap.split_groups == ((0,), (1,), (2,))
    assert len(fm.bridges) == 3


def test_root_with_two_children_is_articulation_point():
    # DFS 根 0 有两个子树：0-1、0-2 且 1-2 无边
    fm = find_fragility(3, [(0, 1), (0, 2)])
    (ap,) = fm.articulation_points
    assert ap.vertex == 0
    assert ap.split_groups == ((1,), (2,))


def test_disconnected_graph_and_isolated_vertices():
    # 分量 {0,1}、{2,3,4} 路径、孤立点 5
    fm = find_fragility(6, [(0, 1), (2, 3), (3, 4)])
    assert fm.component_count == 3
    aps = {ap.vertex: ap.split_groups for ap in fm.articulation_points}
    # 两点分量无割点；路径中点 3 是割点
    assert aps == {3: ((2,), (4,))}
    assert [(b.a, b.b) for b in fm.bridges] == [(0, 1), (2, 3), (3, 4)]


def test_parallel_edges_and_self_loops_ignored():
    # 平行边去重后 (0,1) 是桥；自环不影响连通性
    fm = find_fragility(3, [(0, 1), (1, 0), (0, 0), (1, 2)])
    assert fm.component_count == 1
    assert [(b.a, b.b) for b in fm.bridges] == [(0, 1), (1, 2)]
    assert [ap.vertex for ap in fm.articulation_points] == [1]


def test_single_vertex_no_edges():
    fm = find_fragility(1, [])
    assert fm.component_count == 1
    assert fm.articulation_points == ()
    assert fm.bridges == ()


def test_complete_graph_has_no_fragility():
    edges = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    fm = find_fragility(6, edges)
    assert fm.component_count == 1
    assert fm.articulation_points == ()
    assert fm.bridges == ()


def test_matches_brute_force_randomized():
    rng = random.Random(20260920)
    for _ in range(2000):
        n = rng.randint(1, 10)
        edges = []
        for u in range(n):
            for v in range(u, n):
                if rng.random() < 0.3:
                    edges.append((u, v))
                    if rng.random() < 0.2:
                        edges.append((v, u))  # 平行重复边
        fm = find_fragility(n, edges)

        base_groups = _components_after_removal(n, edges)
        assert fm.component_count == len(base_groups)

        # 割点：移除后分量数增加；分裂组即其原分量被拆出的各组
        expected_aps = {}
        for v in range(n):
            after = _components_after_removal(n, edges, remove_v=v)
            if len(after) > len(base_groups):
                vcomp = next(g for g in base_groups if v in g)
                expected_aps[v] = tuple(
                    sorted(g for g in after if set(g) <= set(vcomp))
                )
        got_aps = {ap.vertex: ap.split_groups for ap in fm.articulation_points}
        assert got_aps == expected_aps

        # 桥：移除后分量数增加的边；两侧为断裂后的两个分量
        norm_edges = {
            (min(a, b), max(a, b)) for a, b in edges if a != b
        }
        expected_bridges = {}
        for e in norm_edges:
            after = _components_after_removal(n, edges, remove_e=e)
            if len(after) > len(base_groups):
                a, b = e
                side_a = next(g for g in after if a in g)
                side_b = next(g for g in after if b in g)
                expected_bridges[e] = (side_a, side_b)
        got_bridges = {
            (b.a, b.b): (b.side_a, b.side_b) for b in fm.bridges
        }
        assert got_bridges == expected_bridges
