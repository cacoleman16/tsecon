import json, re, sys, collections
rs = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")]
ab = [r for r in rs if r["class"] in ("abort", "timeout", "memoryerror", "panic")]
groups = collections.defaultdict(list)
for r in ab:
    st = r.get("stderr", "") or ""
    frames = re.findall(r"\d+: (tsecon\w*::[\w:<>]+)", st)
    frame = frames[0] if frames else ""
    alloc = re.search(r"memory allocation of (\d+) bytes", st)
    key = (r["fn"], r["param"])
    groups[key].append((r["value"], r["class"], frame, alloc.group(1) if alloc else "", r.get("msg", "")[:120], r["wall"]))
for k, v in sorted(groups.items()):
    print(k)
    for e in v:
        print("    ", e)
