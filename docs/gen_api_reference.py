"""Generate docs/reference/api.md from the type stub bindings/python/tsecon.pyi.

The stub is the single source of truth for the callable surface (a pytest
guard keeps it in sync with the compiled module), so the API reference is
generated from it rather than hand-maintained.

    .venv/bin/python docs/gen_api_reference.py
"""
import re
from pathlib import Path

REPO = Path(__file__).parents[1]
PYI = REPO / "bindings" / "python" / "tsecon.pyi"
OUT = REPO / "docs" / "reference" / "api.md"

text = PYI.read_text().splitlines()

# Walk the stub, tracking the most recent "# ---- section ----" comment as a
# group heading, and collect (section, signature, docstring) per def.
section = "General"
entries = []  # (section, name, signature, doc)
i = 0
while i < len(text):
    line = text[i]
    m_sec = re.match(r"#\s*-+\s*(.+?)\s*$", line)
    if m_sec:
        section = m_sec.group(1).strip()
        i += 1
        continue
    m_def = re.match(r"def (\w+)\(", line)
    if m_def:
        name = m_def.group(1)
        # accumulate the (possibly multi-line) signature up to "-> ...:"
        sig_lines = [line]
        while not sig_lines[-1].rstrip().endswith(":"):
            i += 1
            sig_lines.append(text[i])
        signature = "\n".join(sig_lines)
        # docstring
        doc = ""
        if i + 1 < len(text) and text[i + 1].lstrip().startswith('"""'):
            j = i + 1
            body = text[j].lstrip()[3:]
            if body.rstrip().endswith('"""') and len(body.rstrip()) > 3:
                doc = body.rstrip()[:-3].strip()
            else:
                buf = [body]
                j += 1
                while '"""' not in text[j]:
                    buf.append(text[j])
                    j += 1
                buf.append(text[j].split('"""')[0])
                doc = "\n".join(x.rstrip() for x in buf).strip()
        entries.append((section, name, signature, doc))
    i += 1

# Emit grouped markdown.
lines = [
    "# API reference",
    "",
    "The complete callable surface of `tsecon`, generated from the type stub "
    "(`bindings/python/tsecon.pyi`). Every function returns plain NumPy arrays "
    "and dictionaries — no framework objects. For the *why* and *when* of each "
    "method, see the [model cards](README.md) and the "
    "[guide](../guide/README.md).",
    "",
    f"**{len(entries)} functions.**",
    "",
]
by_section = {}
order = []
for sec, name, sig, doc in entries:
    if sec not in by_section:
        by_section[sec] = []
        order.append(sec)
    by_section[sec].append((name, sig, doc))

for sec in order:
    lines.append(f"## {sec}")
    lines.append("")
    for name, sig, doc in by_section[sec]:
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("```python")
        lines.append(sig)
        lines.append("```")
        lines.append("")
        if doc:
            lines.append(doc)
            lines.append("")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT} ({len(entries)} functions, {len(order)} sections)")
