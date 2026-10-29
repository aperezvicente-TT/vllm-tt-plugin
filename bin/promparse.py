#!/usr/bin/env python3
"""Minimal Prometheus text-format reader: samples, sums and histogram quantiles.

Used by the supervisor to turn vLLM's /metrics (``vllm:time_to_first_token_seconds``
histogram, ``vllm:num_requests_waiting`` gauge, ``vllm:spec_decode_*`` counters, the
plugin's ``glm53_*`` series) into the few numbers it re-exports.
"""

from __future__ import annotations

import math
import re

_LINE_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([^\s]+)(?:\s+\d+)?$")
_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse(text: str) -> list[tuple[str, dict[str, str], float]]:
    out: list[tuple[str, dict[str, str], float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2), m.group(3)
        try:
            v = float(value)
        except ValueError:
            continue
        lab = {k: bytes(val, "utf-8").decode("unicode_escape") for k, val in _LABEL_RE.findall(labels or "")}
        out.append((name, lab, v))
    return out


def total(samples, name: str, **match: str) -> float | None:
    """Sum of every sample called ``name`` whose labels include ``match``; None when absent."""
    acc = None
    for n, lab, v in samples:
        if n != name:
            continue
        if any(lab.get(k) != val for k, val in match.items()):
            continue
        acc = (acc or 0.0) + v
    return acc


def histogram_quantile(samples, base: str, q: float) -> float | None:
    """Quantile ``q`` of a Prometheus histogram ``base`` (cumulative ``_bucket`` series summed
    over every other label), linearly interpolated inside the bucket; None without observations."""
    buckets: dict[float, float] = {}
    for n, lab, v in samples:
        if n != base + "_bucket" or "le" not in lab:
            continue
        le = lab["le"]
        key = math.inf if le == "+Inf" else float(le)
        buckets[key] = buckets.get(key, 0.0) + v
    if not buckets:
        return None
    les = sorted(buckets)
    count = buckets[les[-1]]
    if count <= 0:
        return None
    target = q * count
    prev_le, prev_cum = 0.0, 0.0
    for le in les:
        cum = buckets[le]
        if cum >= target:
            if math.isinf(le):
                return prev_le
            if cum == prev_cum:
                return le
            frac = (target - prev_cum) / (cum - prev_cum)
            return prev_le + (le - prev_le) * frac
        prev_le, prev_cum = le, cum
    return les[-1] if not math.isinf(les[-1]) else prev_le


def histogram_mean(samples, base: str) -> float | None:
    s, c = total(samples, base + "_sum"), total(samples, base + "_count")
    if s is None or not c:
        return None
    return s / c
