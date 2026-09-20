"""批次序列风险轨迹：统一阈值重判、逐批状态压缩与相邻批次变迁统计。

统一阈值重判
------------
不重算任何对齐，只取每个配对已落库的两个标量指标——最长连续互补长度
``c`` 与 3′端连续互补长度 ``t``——在请求给定的统一阈值下重判风险等级，
规则与批次计算完全一致：

* ``t >= 3′端阈值``         → high（聚合酶延伸风险，最优先）；
* 否则 ``c >= 连续互补阈值`` → medium（双链结合稳定）；
* 否则                      → low。

轨迹与片段
----------
对全部批次配对键（按引物名称构造，见 compare.pair_key）的并集，逐批形成
present/absent 状态序列：present 状态携带重判风险等级与涉及引物的序列
签名（自二聚体为单序列，异二聚体按键序两序列）。连续相同状态压缩为
**最大连续片段**；相邻两批均 present 但同名引物序列签名不同的位置必为
片段边界，并在新片段上标记 ``sequence_changed_at_start``。

仅保留片段数大于 1 的轨迹（即发生过风险、存在性或序列变化的配对键）；
全程恒定的配对键不返回。相邻批次变迁按风险等级全序
``low < medium < high`` 统计升降、引入（absent→present）与消失
（present→absent）数量。
"""

from __future__ import annotations

from dataclasses import dataclass

from .compare import RISK_ORDER

# 单个批次中一个配对键的状态：
# None 表示 absent；否则为 (重判风险等级, 涉及引物序列签名)
PairState = tuple[str, tuple[str, ...]] | None


def rejudge_risk_level(
    max_complementary_run: int,
    terminal_run: int,
    *,
    continuous_threshold: int,
    three_prime_threshold: int,
) -> str:
    """用已保存的两个标量指标在统一阈值下重判风险等级（规则与批次计算一致）。"""
    if terminal_run >= three_prime_threshold:
        return "high"
    if max_complementary_run >= continuous_threshold:
        return "medium"
    return "low"


@dataclass(frozen=True)
class Segment:
    """轨迹中的一段最大连续相同状态。"""

    start_index: int                  # 起始批次在请求序列中的序号（0 起，含）
    end_index: int                    # 结束批次序号（含）
    present: bool                     # 该片段内配对是否存在于批次中
    risk_level: str | None            # present 时的重判风险等级（absent 为 None）
    sequence_changed_at_start: bool   # 片段起点相对前一批次存在同名引物序列变化


def compress_states(states: list[PairState]) -> list[Segment]:
    """把逐批 present/absent 状态序列压缩为最大连续片段。

    states 按请求批次顺序给出，元素为 None（absent）或
    ``(risk_level, sequences)``。相邻两批均 present 且序列签名不同的
    位置强制成为片段边界，并在新片段标记 ``sequence_changed_at_start``
    （首片段恒为 False）。
    """
    segments: list[Segment] = []
    n = len(states)
    start = 0
    for i in range(1, n + 1):
        if i < n and states[i] == states[start]:
            continue
        current = states[start]
        previous = states[start - 1] if start > 0 else None
        sequence_changed = (
            start > 0
            and previous is not None
            and current is not None
            and previous[1] != current[1]
        )
        segments.append(
            Segment(
                start_index=start,
                end_index=i - 1,
                present=current is not None,
                risk_level=current[0] if current is not None else None,
                sequence_changed_at_start=sequence_changed,
            )
        )
        start = i
    return segments


@dataclass(frozen=True)
class TransitionCounts:
    """相邻两批之间的配对变迁计数。"""

    escalated: int      # 两批均存在且风险等级上升
    deescalated: int    # 两批均存在且风险等级下降
    introduced: int     # 前批不存在、后批存在
    resolved: int       # 前批存在、后批不存在


def count_transition(
    previous: dict[tuple[str, str], PairState],
    current: dict[tuple[str, str], PairState],
) -> TransitionCounts:
    """统计相邻两批（各自 配对键 -> present 状态 的映射）之间的变迁数量。"""
    escalated = deescalated = introduced = resolved = 0
    for key in set(previous) | set(current):
        before = previous.get(key)
        after = current.get(key)
        if before is None:
            introduced += 1
        elif after is None:
            resolved += 1
        elif RISK_ORDER[after[0]] > RISK_ORDER[before[0]]:
            escalated += 1
        elif RISK_ORDER[after[0]] < RISK_ORDER[before[0]]:
            deescalated += 1
    return TransitionCounts(
        escalated=escalated,
        deescalated=deescalated,
        introduced=introduced,
        resolved=resolved,
    )
