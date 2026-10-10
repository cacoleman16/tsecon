import json, sys, collections
path = sys.argv[1] if len(sys.argv) > 1 else "results.jsonl"
rs = [json.loads(l) for l in open(path)]
cnt = collections.Counter(r["class"] for r in rs)
print("cases", len(rs), dict(cnt))
print("callables", len({r["fn"] for r in rs}), "variants", len({r["variant"] for r in rs}),
      "params(fn,param)", len({(r["fn"], r["param"]) for r in rs}))
byval = collections.defaultdict(collections.Counter)
for r in rs:
    byval[r["value"]][r["class"]] += 1
for v, c in byval.items():
    print(" ", v, dict(c))
show = sys.argv[2:] or ["abort", "timeout", "memoryerror", "panic", "unnamed"]
for cl in show:
    print("=" * 20, cl)
    for r in rs:
        if r["class"] == cl:
            m = (r.get("msg") or r.get("stderr") or "").replace("\n", " | ")
            print(f"{r['id']} [{r.get('type','')}] t={r.get('t')} wall={r.get('wall')} rss={r.get('maxrss_mb')} :: {m[:260]}")
