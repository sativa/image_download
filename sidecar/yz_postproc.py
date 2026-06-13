"""yz_postproc — 榆中后处理 thin wrapper(历史脚本, 算法已通用化进 postproc.py).

⚠️ 通用化说明: sliver 清理等标准后处理已抽到区域无关的 `postproc.py`
(eliminate_slivers / fill_gaps_holes / fix_invalid / standardize / run_postproc)。
本文件保留作榆中示例 + 向后兼容(原引用 yz_postproc.eliminate_slivers 仍可用),
直接从通用模块 re-export, 不再重复实现(历史实现见 git 历史 a9019e7 之前的本文件)。

标准末步, 所有矢量成品(SMOOTH/SMOOTH2/SMOOTH3... 或任意新区域)都应跑 postproc.run_postproc。

判据(按"细长度"非纯面积, 避免误删真实小田块):
  - 平均宽度  w = area / perimeter < W_MIN  (默认 2.0m; 线状物 w 很小, 紧凑地块 w 大)
    例: 100m 长 4m 宽的田埂残片 area=400 peri≈208 -> w≈1.9m 判 sliver;
        20m×20m 真田块 area=400 peri=80 -> w=5m 不判.
  - 或 Polsby-Popper 紧凑度 PP = 4πA/P² < PP_MIN(默认0.05) 且 area < A_MAX(默认很小)
    -> 极不紧凑(蜿蜒细线)的小图斑也判 sliver.
处理 = eliminate(并入邻块)不是删:
  - 每个 sliver 并入与它"共享边界最长"的相邻地块(union, 取邻块的 class)-> 保持零重叠/无空白覆盖.
  - 无邻的(truly isolated)保留(定其底层类), 不删(否则会留空白).
  - 迭代多轮(并完可能暴露新 sliver), 直到收敛或达 max_rounds.

在米制 CRS(UTM 32648)下运行(宽度/周长才有物理意义).
"""
import sys as _sys
from pathlib import Path as _Path

_SC = str(_Path(__file__).resolve().parent)
if _SC not in _sys.path:
    _sys.path.insert(0, _SC)

# 向后兼容 re-export: 算法单一真源在区域无关的 postproc.py。
from postproc import (  # noqa: F401
    is_sliver, eliminate_slivers, find_slivers,
    fill_gaps_holes, fix_invalid, standardize, run_postproc,
)

__all__ = ["is_sliver", "eliminate_slivers", "find_slivers",
           "fill_gaps_holes", "fix_invalid", "standardize", "run_postproc"]
