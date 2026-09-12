"""Shared helpers for the round-13 sweeps: round 11's walkers plus paths."""
from __future__ import annotations

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

_spec = importlib.util.spec_from_file_location("r11common", os.path.join(HERE, "..", "round11", "common.py"))
_r11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r11)

API_MD, CARDS, PYI = _r11.API_MD, _r11.CARDS, _r11.PYI
bits_equal, card_for, doc_tokens = _r11.bits_equal, _r11.card_for, _r11.doc_tokens
json_roundtrip, log, nonfinite_paths = _r11.json_roundtrip, _r11.log, _r11.nonfinite_paths
pickle_roundtrip, shapes, top_keys, walk = _r11.pickle_roundtrip, _r11.shapes, _r11.top_keys, _r11.walk

GUIDE = os.path.join(REPO, "docs", "guide")
GUIDE_CHAPTERS = [
    os.path.join(GUIDE, "13-nonlinear-dynamics.md"),
    os.path.join(GUIDE, "14-panel-time-series.md"),
    os.path.join(GUIDE, "15-term-structure.md"),
    os.path.join(GUIDE, "16-lp-vs-var-head-to-head.md"),
]
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

import re as _re

CARD_OF = {
    "setar_threshold_ci": ("cointegration-regime.md", r"^## `setar_threshold_ci`", r"^## `star`"),
    "threshold_var_girf": ("cointegration-regime.md", r"^### Generalized impulse responses — `threshold_var_girf`", r"^## `threshold_var_test`"),
    "var_girf": ("var-svar.md", r"^### Generalized impulse responses — `var_girf`", r"^\*\*References\.\*\* Koop"),
    "jsz_fit": ("term-structure.md", r"^## What it estimates", r"^## References"),
    "jsz_loadings": ("term-structure.md", r"^## What it estimates", r"^## References"),
    "panel_distributed_lag": ("panel.md", r"^## Distributed-lag panel regressions", r"^## LP-DiD"),
}


def card_section(name):
    """The card text that documents `name` (start/end headings in CARD_OF)."""
    fn, start, end = CARD_OF[name]
    text = open(os.path.join(CARDS, fn), encoding="utf-8").read()
    s = _re.search(start, text, _re.M).start()
    e = _re.search(end, text[s + 10:], _re.M)
    return text[s : s + 10 + e.start()] if e else text[s:]


def code_tokens(text):
    """Identifiers inside backticks, tolerant of `girf[h][variable]` and
    `slope_region_low/high` spellings (every identifier in the span counts)."""
    out = set()
    for span in _re.findall(r"`([^`]*)`", text or ""):
        out |= set(_re.findall(r"[A-Za-z_][A-Za-z_0-9]*", span))
    return out
