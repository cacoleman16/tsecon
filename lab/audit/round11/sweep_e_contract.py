"""Sweep E — the result-object contract, for every public callable.

(i)   tsecon.summarize(result).summary() renders;
(ii)  json.dumps(default=...) and pickle round-trip the result;
(iii) every float is finite, or its NaN/inf is mentioned in the docstring;
(iv)  returned keys vs docstring tokens (undocumented keys) and docstring
      key lists vs returned keys (phantom keys — candidates only);
(v)   array shapes are dumped to shapes.json for a manual docstring diff.

Run:  .venv-wt/bin/python lab/audit/round11/sweep_e_contract.py
Out:  lab/audit/round11/out/sweep_e.log, sweep_e.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import (  # noqa: E402
    HERE, bits_equal, card_for, doc_tokens, json_roundtrip, log, nonfinite_paths,
    pickle_roundtrip, shapes, top_keys,
)
from registry import NAMES, build  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def doc_key_candidates(doc):
    """Names the docstring presents as returned keys.

    Two forms are recognised: (a) a 'Keys:' / 'Returns' sentence, in which
    backticked or bare comma-separated identifiers are keys; (b) any backticked
    identifier. Only (a) is used for the phantom check; (b) for the
    undocumented check (a returned key must appear somewhere in the doc)."""
    doc = doc or ""
    flat = re.sub(r"\s+", " ", doc)
    sent = []
    for m in re.finditer(r"(?:Keys:|Returns?(?: a dict with| dict keys:| dict keys| keys:| the)?)\s*(.*?)(?:\.\s|$)", flat):
        sent.append(m.group(1))
    keys = set()
    for s in sent:
        keys |= set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", s))
        # bare identifiers separated by commas / slashes (the "Keys: a, b, c" style)
        bare = re.findall(r"(?<![`\w.])([a-z][a-z0-9_]{1,})(?![`\w(])", s)
        keys |= {b for b in bare if "_" in b}
    return keys


def main():
    fh = open(os.path.join(OUT, "sweep_e.log"), "w")
    report = {}
    n_ok = 0
    for name in NAMES:
        fn = getattr(tsecon, name)
        rec = {"called": False}
        try:
            args, kwargs = build(name, T=200, seed=0)
            res = fn(*args, **kwargs)
            rec["called"] = True
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()[-600:]
            log(fh, f"[{name}] CALL FAILED {rec['error']}")
            report[name] = rec
            continue
        n_ok += 1
        rec["type"] = type(res).__name__
        # (i) summarize
        try:
            text = tsecon.summarize(res, title=name).summary()
            rec["summarize_ok"] = True
            rec["summarize_lines"] = text.count("\n") + 1
        except Exception as exc:  # noqa: BLE001
            rec["summarize_ok"] = False
            rec["summarize_err"] = f"{type(exc).__name__}: {exc}"
            log(fh, f"[{name}] SUMMARIZE FAILED {rec['summarize_err']}")
        # (ii) json + pickle
        ok, why, back = json_roundtrip(res)
        rec["json_ok"] = ok
        if not ok:
            rec["json_err"] = why
            log(fh, f"[{name}] JSON FAILED {why}")
        else:
            # value fidelity: compare decoded to the original structurally
            same, w2 = bits_equal(res if not isinstance(res, np.ndarray) else res.tolist(), back)
            rec["json_faithful"] = same
            if not same:
                rec["json_diff"] = w2
                log(fh, f"[{name}] JSON round-trip differs: {w2}")
        ok, why, pback = pickle_roundtrip(res)
        rec["pickle_ok"] = ok
        if ok:
            same, w2 = bits_equal(res, pback)
            rec["pickle_faithful"] = same
            if not same:
                rec["pickle_diff"] = w2
                log(fh, f"[{name}] PICKLE round-trip differs: {w2}")
        else:
            rec["pickle_err"] = why
            log(fh, f"[{name}] PICKLE FAILED {why}")
        # (iii) non-finite floats
        nf = nonfinite_paths(res)
        rec["nonfinite"] = nf
        doc = fn.__doc__ or ""
        mentions = bool(re.search(r"\bNaN\b|\bnan\b|\binf\b|\binfinit|non-finite|±inf|\+/-inf", doc))
        rec["doc_mentions_nonfinite"] = mentions
        if nf:
            level = "documented" if mentions else "UNDOCUMENTED"
            log(fh, f"[{name}] non-finite ({level}): {nf[:6]}{' ...' if len(nf) > 6 else ''}")
        # (iv) keys
        keys = top_keys(res)
        toks = doc_tokens(doc)
        flat = re.sub(r"\s+", " ", doc)
        undocumented = sorted(k for k in keys if k not in toks and not re.search(rf"\b{re.escape(k)}\b", flat))
        rec["keys"] = sorted(keys)
        rec["undocumented_keys"] = undocumented
        cands = doc_key_candidates(doc)
        phantom = sorted(c for c in cands if c not in keys)
        rec["phantom_candidates"] = phantom
        # model-card keys: backticked identifiers in the card that look like keys
        cards = card_for(name)
        rec["cards"] = [c[0] for c in cards]
        if undocumented:
            log(fh, f"[{name}] UNDOCUMENTED keys (not in __doc__): {undocumented}")
        if phantom:
            log(fh, f"[{name}] phantom candidates (in doc key list, not returned): {phantom}")
        # (v) shapes
        rec["shapes"] = {p: list(s) for p, s in shapes(res).items()}
        report[name] = rec
    log(fh, f"\nREACHED {n_ok}/{len(NAMES)} canonical calls")
    json.dump(report, open(os.path.join(OUT, "sweep_e.json"), "w"), indent=1, default=str)
    fh.close()


if __name__ == "__main__":
    main()
