#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py ── Shared utilities for patrol modules
"""

class PID:
    """Simple PID Controller"""
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
