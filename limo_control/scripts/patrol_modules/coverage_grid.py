#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coverage_grid.py ── 依里程計座標追蹤機器人走過區域

設計理念
--------
• 使用固定網格解析度 (cell_size)，將 odom 座標 (x,y) 量化為離散格 (ix,iy)。  
• 進入新格就記錄到 visited set，藉此估算「已覆蓋格數 / 全部格數」。  
  - 全部格數 = 當前已訪格的包絡盒 (min_ix ~ max_ix, min_iy ~ max_iy) 面積  
• 不需要地圖，可隨時獨立計算；適合初期貼牆階段管控「跑多久就返航」。

API
---
    grid = CoverageGrid(cell_size=0.5)
    grid.mark(x, y)          # 每回合用 odom 更新
    ratio = grid.ratio()     # 0.0 ~ 1.0 粗略覆蓋率
    pct   = grid.percent()   # 百分比
    stats = grid.stats()     # (visited, total, min_ix, max_ix, min_iy, max_iy)
"""
class PID:
    """簡易 PID 控制器"""
    def __init__(self, kp, ki, kd, limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self.i = 0.0
        self.prev = None

    def reset(self):
        self.i = 0.0
        self.prev = None

    def step(self, err, dt):
        p = self.kp * err
        self.i += err * dt
        i = self.ki * self.i
        d = 0.0
        if self.prev is not None and dt > 0:
            d = self.kd * (err - self.prev) / dt
        self.prev = err
        out = p + i + d
        if self.limit is not None:
            out = max(-self.limit, min(self.limit, out))
        return out

import math

__all__ = ["CoverageGrid"]


class CoverageGrid:
    def __init__(self, cell_size: float = 0.5):
        if cell_size <= 0:
            raise ValueError("cell_size must be > 0")
        self.res = cell_size
        self.visited = set()  # {(ix, iy)}
        # 包絡盒
        self.min_ix = self.max_ix = None
        self.min_iy = self.max_iy = None

    # --------------------------------------------------
    def _update_bbox(self, ix, iy):
        if self.min_ix is None:
            self.min_ix = self.max_ix = ix
            self.min_iy = self.max_iy = iy
        else:
            self.min_ix = min(self.min_ix, ix)
            self.max_ix = max(self.max_ix, ix)
            self.min_iy = min(self.min_iy, iy)
            self.max_iy = max(self.max_iy, iy)

    # --------------------------------------------------
    def mark(self, x: float, y: float):
        """依據 (x,y) 數值加入 visited set"""
        ix = int(math.floor(x / self.res))
        iy = int(math.floor(y / self.res))
        if (ix, iy) not in self.visited:
            self.visited.add((ix, iy))
            self._update_bbox(ix, iy)

    # --------------------------------------------------
    def area_cells(self):
        """目前包絡盒的總格數"""
        if self.min_ix is None:
            return 0
        w = self.max_ix - self.min_ix + 1
        h = self.max_iy - self.min_iy + 1
        return w * h

    # --------------------------------------------------
    def visited_cells(self):
        return len(self.visited)

    # --------------------------------------------------
    def ratio(self) -> float:
        """已覆蓋 / 可能區域 (0.0~1.0)"""
        total = self.area_cells()
        return float(len(self.visited)) / total if total else 0.0

    def percent(self) -> float:
        return self.ratio() * 100.0

    # --------------------------------------------------
    def stats(self):
        """回傳 (visited, total, min_ix, max_ix, min_iy, max_iy)"""
        return (
            self.visited_cells(),
            self.area_cells(),
            self.min_ix,
            self.max_ix,
            self.min_iy,
            self.max_iy,
        )

    # --------------------------------------------------
    def reset(self):
        self.visited.clear()
        self.min_ix = self.max_ix = None
        self.min_iy = self.max_iy = None

