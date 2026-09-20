"""跨批次风险轨迹接口端到端测试与片段压缩逻辑单测。"""

from __future__ import annotations

import os

import pytest

from app.trajectory import (
    build_trajectories,
    classify_risk,
    summarize_transitions,
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("trajectories_db")
    os.environ["PRIMER_DB_PATH"] = str(tmp_dir / "test.db")

    from app import db, main

    db.init_db()
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


# 三个按时间排列的批次（提交时原阈值任意，轨迹接口会用统一阈值重判）：
#   B0: F=GGGGGAAAAA, R=GGGGGTTTTT
#   B1: F=GGGGGAAAA（序列变化），R 不变，X=AAAATTTT（新引入）
#   B2: F=GGGGGAAAAA（恢复原序列），R 不变（X 消失）
#
# 已用 dimer 算法核实的两个已保存标量 (最长连续互补 c, 3′端连续 t)：
#   F 自二聚体      (0,0)（两种 F 序列皆然）
#   R 自二聚体      (0,0)
#   X 自二聚体      (8,8)
#   F(10A)|R        (5,5)      F(9A)|R   (4,4)
#   F(9A)|X         (4,4)      R|X       (4,4)
B0 = {
    "primers": [
        {"name": "F", "sequence": "GGGGGAAAAA"},
        {"name": "R", "sequence": "GGGGGTTTTT"},
    ],
    "continuous_threshold": 4,
    "three_prime_threshold": 3,
}
B1 = {
    "primers": [
        {"name": "F", "sequence": "GGGGGAAAA"},
        {"name": "R", "sequence": "GGGGGTTTTT"},
        {"name": "X", "sequence": "AAAATTTT"},
    ],
    "continuous_threshold": 4,
    "three_prime_threshold": 3,
}
B2 = {
    "primers": [
        {"name": "F", "sequence": "GGGGGAAAAA"},
        {"name": "R", "sequence": "GGGGGTTTTT"},
    ],
    "continuous_threshold": 4,
    "three_prime_threshold": 3,
}

PATH = "/api/batch-series/risk-trajectories"


@pytest.fixture(scope="module")
def series_ids(client):
    b0 = client.post("/api/batches", json=B0).json()["id"]
    b1 = client.post("/api/batches", json=B1).json()["id"]
    b2 = client.post("/api/batches", json=B2).json()["id"]
    return [b0, b1, b2]


def _trajectories(client, ids, ct=4, tt=3):
    return client.post(
        PATH,
        json={
            "batch_ids": ids,
            "continuous_threshold": ct,
            "three_prime_threshold": tt,
        },
    )


def _by_key(data):
    return {t["pair_key"]: t for t in data["trajectories"]}


def _assert_segments_cover_axis(trajectory, n):
    """片段必须无缝覆盖批次序号轴 0..n-1。"""
    segments = trajectory["segments"]
    assert segments[0]["start_batch_index"] == 0
    assert segments[-1]["end_batch_index"] == n - 1
    for prev, cur in zip(segments, segments[1:]):
        assert cur["start_batch_index"] == prev["end_batch_index"] + 1
    for seg in segments:
        assert seg["start_batch_index"] <= seg["end_batch_index"]


# ---------- 基本结构与过滤 ----------

def test_trajectory_response_shape_and_filtering(client, series_ids):
    resp = _trajectories(client, series_ids)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["batch_ids"] == series_ids
    assert data["batch_count"] == 3
    assert data["continuous_threshold"] == 4
    assert data["three_prime_threshold"] == 3
    # 全部批次配对键并集：F、R、X 三个自二聚体 + F|R、F|X、R|X
    assert data["pair_key_count"] == 6

    by = _by_key(data)
    # R 自二聚体：全程 present、low、无序列变化 -> 被过滤，不出现在轨迹中
    assert "R" not in by
    # 其余 5 个键均发生存在性或序列变化
    assert data["trajectory_count"] == 5
    assert set(by) == {"F", "X", "F|R", "F|X", "R|X"}

    # 轨迹按配对键升序
    keys = [t["pair_key"] for t in data["trajectories"]]
    assert keys == sorted(keys)

    for trajectory in data["trajectories"]:
        _assert_segments_cover_axis(trajectory, 3)
        assert (
            trajectory["risk_changed"]
            or trajectory["presence_changed"]
            or trajectory["sequence_changed"]
        )


def test_presence_trajectory_segments(client, series_ids):
    data = _trajectories(client, series_ids).json()
    x_self = _by_key(data)["X"]
    assert x_self["is_self_dimer"] is True
    assert x_self["name_a"] == x_self["name_b"] == "X"
    assert x_self["presence_changed"] is True
    assert x_self["risk_changed"] is False
    assert x_self["sequence_changed"] is False
    # absent(0) -> present high(1) -> absent(2)，压成 3 个最大片段
    assert [
        (s["present"], s["risk_level"], s["sequence_changed_at_boundary"])
        for s in x_self["segments"]
    ] == [
        (False, None, False),
        (True, "high", False),   # absent→present 边界不是序列变化边界
        (False, None, False),
    ]
    assert [(s["start_batch_index"], s["end_batch_index"]) for s in x_self["segments"]] == [
        (0, 0), (1, 1), (2, 2)
    ]

    for key in ("F|X", "R|X"):
        traj = _by_key(data)[key]
        assert traj["presence_changed"] is True
        assert traj["is_self_dimer"] is False
        statuses = [(s["present"], s["risk_level"]) for s in traj["segments"]]
        assert statuses == [(False, None), (True, "high"), (False, None)]


def test_sequence_change_marks_segment_boundaries(client, series_ids):
    data = _trajectories(client, series_ids).json()
    by = _by_key(data)

    # F 自二聚体：全程 present 且始终 low，仅因序列变化入选；
    # B0→B1 与 B1→B2 两次序列变化都必须切开片段并在边界标记
    f_self = by["F"]
    assert f_self["risk_changed"] is False
    assert f_self["presence_changed"] is False
    assert f_self["sequence_changed"] is True
    assert len(f_self["segments"]) == 3
    assert [s["sequence_changed_at_boundary"] for s in f_self["segments"]] == [
        False, True, True
    ]
    assert all(s["present"] and s["risk_level"] == "low" for s in f_self["segments"])

    # F|R：统一阈值 (4,3) 下三批均 high（(5,5) 与 (4,4) 的 t 都 ≥3），
    # 风险不变，但 F 序列变化仍在两条边界标记
    fr = by["F|R"]
    assert fr["risk_changed"] is False
    assert fr["presence_changed"] is False
    assert fr["sequence_changed"] is True
    assert len(fr["segments"]) == 3
    assert all(s["risk_level"] == "high" for s in fr["segments"])
    assert [s["sequence_changed_at_boundary"] for s in fr["segments"]] == [
        False, True, True
    ]


def test_transition_summary(client, series_ids):
    data = _trajectories(client, series_ids).json()
    assert data["transitions"] == [
        {
            "index": 0,
            "escalated_count": 0,
            "deescalated_count": 0,
            "introduced_count": 3,        # X 自二聚体、F|X、R|X
            "disappeared_count": 0,
            "sequence_changed_count": 2,  # F 自二聚体、F|R
        },
        {
            "index": 1,
            "escalated_count": 0,
            "deescalated_count": 0,
            "introduced_count": 0,
            "disappeared_count": 3,
            "sequence_changed_count": 2,
        },
    ]


# ---------- 统一阈值重判（不重算对齐） ----------

def test_uniform_threshold_rejudges_risk_without_realignment(client, series_ids):
    # ct=5、tt=6：F|R 在 B0/B2 为 (5,5)->medium，B1 为 (4,4)->low，
    # 序列变化同时发生；X 自二聚体 (8,8) 仍 high。
    data = _trajectories(client, series_ids, ct=5, tt=6).json()
    by = _by_key(data)

    fr = by["F|R"]
    assert fr["risk_changed"] is True
    assert fr["sequence_changed"] is True
    statuses = [
        (s["present"], s["risk_level"], s["sequence_changed_at_boundary"])
        for s in fr["segments"]
    ]
    assert statuses == [
        (True, "medium", False),
        (True, "low", True),
        (True, "medium", True),
    ]

    # F|X、R|X 在 B1 的 (4,4) 于 (5,6) 下为 low，但仍因存在性变化入选
    assert by["F|X"]["segments"][1]["risk_level"] == "low"
    assert by["X"]["segments"][1]["risk_level"] == "high"

    assert data["transitions"] == [
        {
            "index": 0,
            "escalated_count": 0,
            "deescalated_count": 1,        # F|R medium -> low
            "introduced_count": 3,
            "disappeared_count": 0,
            "sequence_changed_count": 2,
        },
        {
            "index": 1,
            "escalated_count": 1,          # F|R low -> medium
            "deescalated_count": 0,
            "introduced_count": 0,
            "disappeared_count": 3,
            "sequence_changed_count": 2,
        },
    ]


def test_deterministic_output(client, series_ids):
    a = _trajectories(client, series_ids).json()
    b = _trajectories(client, series_ids).json()
    assert a == b


def test_read_only_does_not_modify_batches(client, series_ids):
    before = [client.get(f"/api/batches/{bid}").json() for bid in series_ids]
    assert _trajectories(client, series_ids).status_code == 200
    after = [client.get(f"/api/batches/{bid}").json() for bid in series_ids]
    assert before == after


def test_no_change_series_returns_empty_trajectories(client):
    # 两个内容完全一致的批次：无任何变化轨迹，但仍返回零值相邻汇总
    bid = client.post("/api/batches", json=B0).json()["id"]
    other = client.post("/api/batches", json=B0).json()["id"]
    resp = _trajectories(client, [bid, other])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["trajectory_count"] == 0
    assert data["trajectories"] == []
    assert data["pair_key_count"] == 3
    assert data["transitions"] == [
        {
            "index": 0,
            "escalated_count": 0,
            "deescalated_count": 0,
            "introduced_count": 0,
            "disappeared_count": 0,
            "sequence_changed_count": 0,
        }
    ]


# ---------- 错误码 ----------

def test_missing_batch_returns_404(client, series_ids):
    missing = "no-such-batch-id"
    assert _trajectories(client, [missing, series_ids[1]]).status_code == 404
    assert _trajectories(client, [series_ids[0], missing]).status_code == 404
    assert _trajectories(client, [missing, missing]).status_code == 422  # 重复优先 422


def test_duplicate_batch_ids_return_422(client, series_ids):
    b0, b1, _ = series_ids
    resp = _trajectories(client, [b0, b1, b0])
    assert resp.status_code == 422
    assert "重复" in resp.json()["detail"]

    # 同一批次出现两次也算重复
    assert _trajectories(client, [b0, b0]).status_code == 422


@pytest.mark.parametrize(
    "ids,ct,tt",
    [
        ("too_few", 4, 3),        # 仅 1 个批次
        ("too_many", 4, 3),       # 21 个批次
        ("ok", 0, 3),             # 连续阈值非法
        ("ok", 4, 0),             # 3′端阈值非法
        ("ok", 201, 3),           # 连续阈值超上限
        ("ok", 4, 201),           # 3′端阈值超上限
    ],
)
def test_invalid_request_returns_422(client, series_ids, ids, ct, tt):
    if ids == "too_few":
        batch_ids = [series_ids[0]]
    elif ids == "too_many":
        batch_ids = [series_ids[0]] * 21
    else:
        batch_ids = series_ids[:2]
    resp = client.post(
        PATH,
        json={"batch_ids": batch_ids, "continuous_threshold": ct,
              "three_prime_threshold": tt},
    )
    assert resp.status_code == 422, resp.text


def test_empty_batch_id_string_returns_422(client, series_ids):
    resp = client.post(
        PATH,
        json={
            "batch_ids": ["  ", series_ids[0]],
            "continuous_threshold": 4,
            "three_prime_threshold": 3,
        },
    )
    assert resp.status_code == 422


# ---------- 片段压缩逻辑单测 ----------

def test_classify_risk_rule():
    assert classify_risk(5, 0, continuous_threshold=4, three_prime_threshold=3) == "medium"
    assert classify_risk(3, 3, continuous_threshold=4, three_prime_threshold=3) == "high"
    assert classify_risk(4, 3, continuous_threshold=4, three_prime_threshold=3) == "high"
    assert classify_risk(3, 2, continuous_threshold=4, three_prime_threshold=3) == "low"


def test_build_trajectories_compresses_maximal_runs():
    states = {
        # 全程 present、风险与序列均无变化 -> 单片段、被过滤
        ("S", "S"): ["low", "low", "low", "low"],
        # present high ×2 -> absent ×1 -> present medium：存在性 + 风险变化
        ("A", "B"): ["high", "high", None, "medium"],
        # 全程 present：风险恒 high，但边界 2 有序列变化
        ("A", "C"): ["high", "high", "high", "high"],
        # 全程 absent：无变化，被过滤
        ("Z", "Z"): [None, None, None, None],
    }
    seq = {(2, ("A", "C")): True}
    trajectories = build_trajectories(states, sequence_changes=seq)
    by = {t.pair_key: t for t in trajectories}

    assert set(by) == {"A|B", "A|C"}
    assert [t.pair_key for t in trajectories] == ["A|B", "A|C"]

    ab = by["A|B"]
    assert ab.presence_changed is True and ab.risk_changed is True
    assert ab.sequence_changed is False
    assert [
        (s.start, s.end, s.present, s.risk_level, s.sequence_changed_at_boundary)
        for s in ab.segments
    ] == [
        (0, 1, True, "high", False),
        (2, 2, False, None, False),
        (3, 3, True, "medium", False),
    ]

    ac = by["A|C"]
    assert ac.sequence_changed is True
    assert [
        (s.start, s.end, s.present, s.risk_level, s.sequence_changed_at_boundary)
        for s in ac.segments
    ] == [
        (0, 1, True, "high", False),
        (2, 3, True, "high", True),
    ]


def test_build_trajectories_sequence_change_requires_both_present():
    # 边界 1 为 absent→present；即便此处有“序列变化”标记，也不应记录
    # （序列变化只在两批都 present 时才有定义，由服务层保证不写该键）
    states = {("A", "B"): [None, "low", "low"]}
    trajectories = build_trajectories(states, sequence_changes={})
    ab, = trajectories
    assert ab.presence_changed is True
    assert ab.risk_changed is False and ab.sequence_changed is False
    assert [
        (s.start, s.end, s.present, s.sequence_changed_at_boundary)
        for s in ab.segments
    ] == [(0, 0, False, False), (1, 2, True, False)]


def test_summarize_transitions_counts():
    states = {
        ("A", "A"): ["low", "medium", "high"],     # 升、升
        ("B", "B"): ["high", None, "low"],         # 消失、引入（下降不计）
        ("C", "D"): ["medium", "low", "low"],      # 降
        ("E", "E"): ["low", "low", "low"],         # 无变化
    }
    seq = {(1, ("A", "A")): True}
    transitions = summarize_transitions(states, sequence_changes=seq, batch_count=3)
    assert transitions == [
        {
            "index": 0,
            "escalated_count": 1,                  # A: low→medium
            "deescalated_count": 1,                # C|D: medium→low
            "introduced_count": 0,
            "disappeared_count": 1,                # B 消失
            "sequence_changed_count": 1,
        },
        {
            "index": 1,
            "escalated_count": 1,                  # A: medium→high
            "deescalated_count": 0,
            "introduced_count": 1,                 # B 重新出现
            "disappeared_count": 0,
            "sequence_changed_count": 0,
        },
    ]
