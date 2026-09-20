"""跨批次风险轨迹：present/absent 序列压缩与相邻批次变化汇总。

对一批**按时间排列**的批次（2～20 个，不可重复），用请求给定的统一阈值，
仅依据各批次配对**已落库的两个标量指标**（最长连续互补 ``c``、3′端连续
互补 ``t``）按现有判定规则重判风险，**不重算对齐**：

* ``t >= 3′端阈值``         → high；
* 否则 ``c >= 连续互补阈值`` → medium；
* 否则                      → low。

按引物名称构造稳定配对键（与 :mod:`app.compare` 一致）：自二聚体保留单名，
异二聚体两名按字典序排列。对全部配对键逐批形成长度等于批次数的
present（配对两端引物均在该批次中）/absent 序列，并把序列压缩为**最大连续
片段**：present/absent 翻转、相邻 present 风险等级变化、同名引物序列变化
（仅当配对在相邻两批中都 present 时可判定）均强制断开片段。

仅返回发生以下任一变化的轨迹：风险等级变化（``risk_changed``）、存在性
变化（``presence_changed``）、同名引物序列变化（``sequence_changed``）。

另对每对相邻批次汇总：等级上升 escalated、下降 deescalated、引入
introduced（前 absent 后 present）、消失 disappeared（前 present 后
absent）与序列变化数量。全部输出按配对键确定性排序。
"""

from __future__ import annotations

from dataclasses import dataclass

# 风险等级全序：low < medium < high
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def classify_risk(
    continuous_run: int,
    three_prime_run: int,
    *,
    continuous_threshold: int,
    three_prime_threshold: int,
) -> str:
    """按统一阈值用两个已保存标量重判风险（与批次计算规则一致）。"""
    if three_prime_run >= three_prime_threshold:
        return "high"
    if continuous_run >= continuous_threshold:
        return "medium"
    return "low"


@dataclass(frozen=True)
class Segment:
    """轨迹上的最大连续片段（批次序号闭区间 [start, end]）。"""

    start: int                      # 起始批次序号（0 起）
    end: int                        # 结束批次序号（含）
    present: bool                   # 片段内配对是否存在
    risk_level: str | None          # present 时为 high/medium/low，absent 时为 None
    sequence_changed_at_boundary: bool = (
        # true 表示与上一片段的分界处（即 start 批次相对 start-1 批次）
        # 发生了同名引物序列变化；仅 present 片段可能为 true
        False
    )


@dataclass(frozen=True)
class Trajectory:
    """一个稳定配对键跨批次的压缩轨迹。"""

    name_a: str                     # 配对键第一名称（异二聚体取字典序较小者）
    name_b: str                     # 配对键第二名称（自二聚体与 name_a 相同）
    is_self_dimer: bool
    segments: tuple[Segment, ...]

    @property
    def pair_key(self) -> str:
        """配对键的可读形式：自二聚体为单名，异二聚体为 ``A|B``。"""
        if self.is_self_dimer:
            return self.name_a
        return f"{self.name_a}|{self.name_b}"

    @property
    def presence_changed(self) -> bool:
        """存在性是否发生过变化（present 与 absent 均出现过）。"""
        return any(seg.present for seg in self.segments) and any(
            not seg.present for seg in self.segments
        )

    @property
    def risk_changed(self) -> bool:
        """各 present 片段的风险等级是否不止一种。"""
        levels = {seg.risk_level for seg in self.segments if seg.present}
        return len(levels) > 1

    @property
    def sequence_changed(self) -> bool:
        """是否存在同名引物序列变化的片段边界。"""
        return any(seg.sequence_changed_at_boundary for seg in self.segments)

    @property
    def has_any_change(self) -> bool:
        """是否命中“风险/存在性/序列变化”任一过滤条件。"""
        return (
            self.risk_changed
            or self.presence_changed
            or self.sequence_changed
        )


def build_trajectories(
    states: dict[tuple[str, str], list[str | None]],
    *,
    sequence_changes: dict[tuple[int, tuple[str, str]], bool],
) -> list[Trajectory]:
    """把逐批状态序列压缩为片段，并过滤无变化轨迹。

    ``states[key][i]`` 为配对键在第 i 批的状态：``None`` 表示 absent，
    否则为统一阈值下重判的风险等级（high/medium/low）。``states`` 中
    每个键的状态序列长度必须相同（等于批次数）。

    ``sequence_changes[(i, key)]`` 为真表示第 i-1 批 → 第 i 批的边界上，
    该配对（两批都 present）中任一同名引物序列发生变化。

    返回轨迹按配对键（名称元组字典序）升序；完全无变化的轨迹不返回。
    """
    trajectories: list[Trajectory] = []
    for key in sorted(states):
        levels = states[key]
        n = len(levels)
        segments: list[Segment] = []
        start = 0
        for i in range(1, n + 1):
            cut = i == n
            if not cut:
                prev, cur = levels[i - 1], levels[i]
                if (prev is None) != (cur is None):
                    cut = True  # present/absent 翻转
                elif prev is not None and cur is not None:
                    if prev != cur or sequence_changes.get((i, key)):
                        cut = True  # 风险等级变化或同名引物序列变化
            if cut:
                end = i - 1
                present = levels[start] is not None
                changed_here = bool(
                    present
                    and start > 0
                    and sequence_changes.get((start, key))
                )
                segments.append(
                    Segment(
                        start=start,
                        end=end,
                        present=present,
                        risk_level=levels[start] if present else None,
                        sequence_changed_at_boundary=changed_here,
                    )
                )
                start = i

        name_a, name_b = key
        trajectory = Trajectory(
            name_a=name_a,
            name_b=name_b,
            is_self_dimer=name_a == name_b,
            segments=tuple(segments),
        )
        if trajectory.has_any_change:
            trajectories.append(trajectory)
    return trajectories


def summarize_transitions(
    states: dict[tuple[str, str], list[str | None]],
    *,
    sequence_changes: dict[tuple[int, tuple[str, str]], bool],
    batch_count: int,
) -> list[dict]:
    """汇总每对相邻批次的升降、引入、消失与序列变化数量。

    返回长度 ``batch_count - 1`` 的列表，第 k 项对应
    ``batch[k] → batch[k+1]``（``index`` 为 0 起的相邻对序号）。
    """
    transitions: list[dict] = []
    for i in range(1, batch_count):
        escalated = deescalated = 0
        introduced = disappeared = changed = 0
        for key, levels in states.items():
            prev, cur = levels[i - 1], levels[i]
            if prev is None and cur is not None:
                introduced += 1
            elif prev is not None and cur is None:
                disappeared += 1
            elif prev is not None and cur is not None:
                if RISK_ORDER[cur] > RISK_ORDER[prev]:
                    escalated += 1
                elif RISK_ORDER[cur] < RISK_ORDER[prev]:
                    deescalated += 1
                if sequence_changes.get((i, key)):
                    changed += 1
        transitions.append(
            {
                "index": i - 1,
                "escalated_count": escalated,
                "deescalated_count": deescalated,
                "introduced_count": introduced,
                "disappeared_count": disappeared,
                "sequence_changed_count": changed,
            }
        )
    return transitions
