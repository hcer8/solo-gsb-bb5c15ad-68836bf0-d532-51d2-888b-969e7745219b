"""批次序列风险轨迹接口端到端测试与轨迹压缩/变迁统计单测。"""

from __future__ import annotations

import os

import pytest

from app.trajectory import (
    compress_states,
    count_transition,
    rejudge_risk_level,
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("trajectory_db")
    os.environ["PRIMER_DB_PATH"] = str(tmp_dir / "test.db")

    from app import db, main

    db.init_db()
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


# 各配对指标已经 dimer 算法核实（ct=4, tt=3 落库）：
#   F|R1: c=5,t=5   F|R2: c=4,t=4   F|R3: c=3,t=3
#   全部自二聚体与 R1|R3: c=0,t=0
F = "GGGGGAAAAA"
R1 = "GGGGGTTTTT"
R2 = "GGGGGGTTTT"
R3 = "GGGGGGGTTT"

THRESHOLDS = {"continuous_threshold": 4, "three_prime_threshold": 3}


def _create_batch(client, primers):
    resp = client.post(
        "/api/batches",
        json={
            "primers": [{"name": n, "sequence": s} for n, s in primers],
            **THRESHOLDS,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


@pytest.fixture(scope="module")
def series_ids(client):
    """风险与序列变化序列：F 恒定，R 逐批缩短 3′端互补。"""
    return [
        _create_batch(client, [("F", F), ("R", R1)]),
        _create_batch(client, [("F", F), ("R", R2)]),
        _create_batch(client, [("F", F), ("R", R3)]),
    ]


@pytest.fixture(scope="module")
def presence_ids(client):
    """存在性变化序列：R 在中间两批消失、末批回归。"""
    return [
        _create_batch(client, [("F", F), ("R", R1)]),
        _create_batch(client, [("F", F)]),
        _create_batch(client, [("F", F)]),
        _create_batch(client, [("F", F), ("R", R1)]),
    ]


def _trajectories(client, batch_ids, ct=4, tt=4):
    return client.post(
        "/api/batch-series/risk-trajectories",
        json={
            "batch_ids": batch_ids,
            "continuous_threshold": ct,
            "three_prime_threshold": tt,
        },
    )


# ---------- 风险与序列变化轨迹 ----------

def test_risk_and_sequence_change_trajectories(client, series_ids):
    resp = _trajectories(client, series_ids)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["batch_ids"] == series_ids
    assert data["batch_count"] == 3
    assert data["continuous_threshold"] == 4
    assert data["three_prime_threshold"] == 4
    # 配对键并集：F、R 自二聚体 + F|R
    assert data["pair_key_count"] == 3
    # F 自二聚体全程 low 且序列未变（单片段），不返回
    assert data["changed_pair_key_count"] == 2

    # 按配对键名称字典序：F|R 在 R 之前
    assert [t["pair_key"] for t in data["trajectories"]] == ["F|R", "R"]

    fr = data["trajectories"][0]
    assert fr["name_a"] == "F" and fr["name_b"] == "R"
    assert fr["is_self_dimer"] is False
    assert fr["segment_count"] == 3
    # 统一阈值 ct=4/tt=4 下重判：t=5→high，t=4→high，t=3→low
    assert fr["segments"] == [
        {"start_index": 0, "end_index": 0, "present": True,
         "risk_level": "high", "sequence_changed_at_start": False},
        {"start_index": 1, "end_index": 1, "present": True,
         "risk_level": "high", "sequence_changed_at_start": True},
        {"start_index": 2, "end_index": 2, "present": True,
         "risk_level": "low", "sequence_changed_at_start": True},
    ]

    r_self = data["trajectories"][1]
    assert r_self["is_self_dimer"] is True
    assert r_self["name_a"] == r_self["name_b"] == "R"
    # R 自二聚体始终 low，但序列逐批变化 → 每批各成一段
    assert [s["risk_level"] for s in r_self["segments"]] == ["low"] * 3
    assert [s["sequence_changed_at_start"] for s in r_self["segments"]] == [
        False, True, True,
    ]


def test_transition_summary_counts(client, series_ids):
    data = _trajectories(client, series_ids).json()
    transitions = data["transitions"]
    assert len(transitions) == 2

    first, second = transitions
    assert first["from_index"] == 0 and first["to_index"] == 1
    assert first["from_batch_id"] == series_ids[0]
    assert first["to_batch_id"] == series_ids[1]
    # 0→1：F|R high→high、自二聚体均 low→low，无变迁
    assert (first["escalated_count"], first["deescalated_count"],
            first["introduced_count"], first["resolved_count"]) == (0, 0, 0, 0)
    # 1→2：仅 F|R high→low
    assert (second["escalated_count"], second["deescalated_count"],
            second["introduced_count"], second["resolved_count"]) == (0, 1, 0, 0)


def test_unified_threshold_rejudges_saved_results(client, series_ids):
    # 落库时 F|R 在全部三批均为 high（ct=4, tt=3）；
    # 统一阈值 ct=6/tt=6 下 c,t 最大仅 5 → 全部重判为 low
    data = _trajectories(client, series_ids, ct=6, tt=6).json()
    fr = next(t for t in data["trajectories"] if t["pair_key"] == "F|R")
    assert [s["risk_level"] for s in fr["segments"]] == ["low"] * 3
    # 风险不再变化，但序列变化仍形成片段边界
    assert [s["sequence_changed_at_start"] for s in fr["segments"]] == [
        False, True, True,
    ]
    # 无风险升降与引入/消失
    for transition in data["transitions"]:
        assert transition["escalated_count"] == 0
        assert transition["deescalated_count"] == 0
        assert transition["introduced_count"] == 0
        assert transition["resolved_count"] == 0


def test_constant_pairs_excluded(client, series_ids):
    data = _trajectories(client, series_ids).json()
    keys = [t["pair_key"] for t in data["trajectories"]]
    # F 自二聚体 c=0,t=0 且序列恒定 → 单片段，不返回
    assert "F" not in keys
    for trajectory in data["trajectories"]:
        assert trajectory["segment_count"] == len(trajectory["segments"]) >= 2


# ---------- 存在性变化轨迹 ----------

def test_presence_trajectory_and_absent_compression(client, presence_ids):
    resp = _trajectories(client, presence_ids)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["pair_key_count"] == 3
    assert [t["pair_key"] for t in data["trajectories"]] == ["F|R", "R"]

    fr = data["trajectories"][0]
    # present → absent（两批压缩为一段）→ present
    assert fr["segments"] == [
        {"start_index": 0, "end_index": 0, "present": True,
         "risk_level": "high", "sequence_changed_at_start": False},
        {"start_index": 1, "end_index": 2, "present": False,
         "risk_level": None, "sequence_changed_at_start": False},
        {"start_index": 3, "end_index": 3, "present": True,
         "risk_level": "high", "sequence_changed_at_start": False},
    ]

    transitions = data["transitions"]
    assert len(transitions) == 3
    # 0→1：F|R 与 R 自二聚体消失
    assert transitions[0]["resolved_count"] == 2
    assert transitions[0]["introduced_count"] == 0
    # 1→2：两批引物完全相同，无任何变迁
    assert (transitions[1]["escalated_count"], transitions[1]["deescalated_count"],
            transitions[1]["introduced_count"], transitions[1]["resolved_count"]
            ) == (0, 0, 0, 0)
    # 2→3：F|R 与 R 自二聚体重新引入
    assert transitions[2]["introduced_count"] == 2
    assert transitions[2]["resolved_count"] == 0


def test_escalation_and_introduced_counts(client):
    # 第一批只有 F、R3（F|R3 重判 low），第二批加入 R1（F|R1 重判 high）
    b1 = _create_batch(client, [("F", F), ("R", R3)])
    b2 = _create_batch(client, [("F", F), ("R", R3), ("X", R1)])
    data = _trajectories(client, [b1, b2]).json()

    transition = data["transitions"][0]
    # F|R：R3→R3 不变；X 自二聚体与 F|X、R|X 为新增
    introduced_keys = {"X", "F|X", "R|X"}
    assert transition["introduced_count"] == len(introduced_keys)
    keys = [t["pair_key"] for t in data["trajectories"]]
    assert set(introduced_keys) <= set(keys)

    # R 序列 R3→R3 未变：F|R 与 R 自二聚体均为单片段，不返回
    assert "F|R" not in keys
    assert "R" not in keys


def test_escalation_between_present_pairs(client):
    # 同名 R 序列变化使 F|R 在统一阈值下升级：R3（t=3）→ R1（t=5）
    b1 = _create_batch(client, [("F", F), ("R", R3)])
    b2 = _create_batch(client, [("F", F), ("R", R1)])
    data = _trajectories(client, [b1, b2]).json()

    transition = data["transitions"][0]
    assert transition["escalated_count"] == 1   # F|R low→high
    assert transition["deescalated_count"] == 0

    fr = next(t for t in data["trajectories"] if t["pair_key"] == "F|R")
    assert [s["risk_level"] for s in fr["segments"]] == ["low", "high"]
    assert fr["segments"][1]["sequence_changed_at_start"] is True


# ---------- 确定性与只读保证 ----------

def test_deterministic(client, series_ids):
    assert _trajectories(client, series_ids).json() == \
        _trajectories(client, series_ids).json()


def test_no_alignment_recompute(client, series_ids, monkeypatch):
    import app.services as services

    calls = []
    original = services.analyze_pair

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(services, "analyze_pair", spy)
    resp = _trajectories(client, series_ids)
    assert resp.status_code == 200
    assert calls == []


def test_batches_not_modified(client, series_ids):
    before = [client.get(f"/api/batches/{bid}").json() for bid in series_ids]
    listing_before = client.get("/api/batches").json()["total"]

    resp = _trajectories(client, series_ids)
    assert resp.status_code == 200

    after = [client.get(f"/api/batches/{bid}").json() for bid in series_ids]
    assert after == before
    assert client.get("/api/batches").json()["total"] == listing_before


# ---------- 错误码 ----------

def test_missing_batch_404(client, series_ids):
    resp = _trajectories(client, [series_ids[0], "no-such-batch", series_ids[1]])
    assert resp.status_code == 404
    assert "不存在" in resp.json()["detail"]


def test_duplicate_batch_ids_422(client, series_ids):
    resp = _trajectories(client, [series_ids[0], series_ids[1], series_ids[0]])
    assert resp.status_code == 422
    assert "重复" in resp.json()["detail"]


def test_invalid_thresholds_422(client, series_ids):
    assert _trajectories(client, series_ids, ct=0).status_code == 422
    assert _trajectories(client, series_ids, tt=0).status_code == 422
    assert _trajectories(client, series_ids, ct=201).status_code == 422
    assert _trajectories(client, series_ids, tt=201).status_code == 422


def test_batch_id_count_bounds_422(client, series_ids):
    # 少于 2 个
    assert _trajectories(client, series_ids[:1]).status_code == 422
    assert _trajectories(client, []).status_code == 422
    # 多于 20 个
    resp = _trajectories(client, [series_ids[0]] * 21)
    assert resp.status_code == 422
    # 空批次 ID
    resp = client.post(
        "/api/batch-series/risk-trajectories",
        json={
            "batch_ids": [series_ids[0], "  "],
            "continuous_threshold": 4,
            "three_prime_threshold": 4,
        },
    )
    assert resp.status_code == 422


# ---------- 轨迹压缩与变迁统计单测 ----------

def test_rejudge_risk_level_rules():
    # high 优先于 medium；均为 >= 语义
    assert rejudge_risk_level(9, 3, continuous_threshold=4, three_prime_threshold=3) == "high"
    assert rejudge_risk_level(4, 2, continuous_threshold=4, three_prime_threshold=3) == "medium"
    assert rejudge_risk_level(3, 2, continuous_threshold=4, three_prime_threshold=3) == "low"
    assert rejudge_risk_level(0, 0, continuous_threshold=1, three_prime_threshold=1) == "low"


def test_compress_states_merges_maximal_runs():
    low_a = ("low", ("AAAA",))
    low_b = ("low", ("CCCC",))
    high_a = ("high", ("AAAA",))
    segments = compress_states([low_a, low_a, None, None, high_a, low_b, low_b])
    assert [(s.start_index, s.end_index) for s in segments] == [
        (0, 1), (2, 3), (4, 4), (5, 6),
    ]
    assert [s.present for s in segments] == [True, False, True, True]
    assert [s.risk_level for s in segments] == ["low", None, "high", "low"]
    # 序列签名变化（low_a→low_b）强制成界并标记；absent→present 不标记
    assert [s.sequence_changed_at_start for s in segments] == [
        False, False, False, True,
    ]


def test_compress_states_constant_single_segment():
    state = ("medium", ("ACGT", "TGCA"))
    segments = compress_states([state, state, state])
    assert len(segments) == 1
    assert segments[0].start_index == 0 and segments[0].end_index == 2
    assert segments[0].sequence_changed_at_start is False


def test_count_transition_categories():
    previous = {
        ("A", "B"): ("low", ("1", "2")),
        ("C", "C"): ("high", ("3",)),
        ("D", "E"): ("medium", ("4", "5")),
        ("F", "G"): ("low", ("6", "7")),
    }
    current = {
        ("A", "B"): ("high", ("1", "2")),   # 升级
        ("C", "C"): ("low", ("3",)),        # 降级
        ("D", "E"): ("medium", ("4", "5")),  # 不变
        ("H", "H"): ("low", ("8",)),        # 引入
    }
    counts = count_transition(previous, current)
    assert counts.escalated == 1
    assert counts.deescalated == 1
    assert counts.introduced == 1
    assert counts.resolved == 1             # F|G 消失
