# Export a results table to LaTeX or Markdown

Every `tsecon` estimator returns a plain `dict` of arrays. That is the contract,
and it is what makes export trivial: `tsecon.summarize` renders any result for
the console, and pandas turns the same arrays into a paper table. No adapters,
no result-object API to learn.

## Step 1: read it in the console

```python
import numpy as np, tsecon

rng, n = np.random.default_rng(0), 300
shock = rng.standard_normal(n)
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.6 * y[t - 1] + 0.8 * shock[t] + 0.5 * rng.standard_normal()

out = tsecon.lp(y, shock, horizons=6, n_lag_controls=4)
print(tsecon.summarize(out, title="Local projection IRF"))
```

```
====================================================================
Local projection IRF — generic Result  (3 fields)
====================================================================
--------------------------------------------------------------------
horizons  (7,)  [0, 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000]
--------------------------------------------------------------------
irf  (7,)  [0.806068, 0.517617, 0.367723, 0.206718, 0.144407, 0.098322, 0.104963]
--------------------------------------------------------------------
se  (7,)  [0.026261, 0.054162, 0.061329, 0.069855, 0.067696, 0.070991, 0.071620]
====================================================================
```

`tsecon.summarize` works on the output of **every** function in the library —
`summarize(tsecon.adf(y))`, `summarize(tsecon.garch_fit(r))`, anything. A
bespoke results object comes back unchanged with its own `.summary()`; a plain
dict becomes a generic `Result` with the aligned dump above. It is still a
`dict` subclass, so nothing downstream breaks.

## Step 2: build the table

```python
import pandas as pd

df = pd.DataFrame({"h": np.asarray(out["horizons"], dtype=int),
                   "irf": out["irf"], "se": out["se"]})
df["lower90"] = df["irf"] - 1.645 * df["se"]
df["upper90"] = df["irf"] + 1.645 * df["se"]
print(df.to_string(index=False, float_format="%.3f"))
```

```
 h   irf    se  lower90  upper90
 0 0.806 0.026    0.763    0.849
 1 0.518 0.054    0.429    0.607
 2 0.368 0.061    0.267    0.469
 3 0.207 0.070    0.092    0.322
 4 0.144 0.068    0.033    0.256
 5 0.098 0.071   -0.018    0.215
 6 0.105 0.072   -0.013    0.223
```

## Step 3: LaTeX

```python
print(df.to_latex(index=False, float_format="%.3f", position="ht",
                  caption="Impulse response to a one-unit shock", label="tab:irf"))
```

```
\begin{table}[ht]
\caption{Impulse response to a one-unit shock}
\label{tab:irf}
\begin{tabular}{rrrrr}
\toprule
h & irf & se & lower90 & upper90 \\
\midrule
0 & 0.806 & 0.026 & 0.763 & 0.849 \\
1 & 0.518 & 0.054 & 0.429 & 0.607 \\
2 & 0.368 & 0.061 & 0.267 & 0.469 \\
3 & 0.207 & 0.070 & 0.092 & 0.322 \\
4 & 0.144 & 0.068 & 0.033 & 0.256 \\
5 & 0.098 & 0.071 & -0.018 & 0.215 \\
6 & 0.105 & 0.072 & -0.013 & 0.223 \\
\bottomrule
\end{tabular}
\end{table}
```

That is `booktabs` output, so `\usepackage{booktabs}` must be in your preamble.
`float_format` controls rounding; tie it to the magnitude of your standard
errors rather than reaching for `%.3f` reflexively.

## Step 4: Markdown, with no extra dependency

`DataFrame.to_markdown()` needs the optional `tabulate` package. Four lines
avoid the dependency:

```python
def to_md(frame, nd=3):                                  # markdown, no extra dependency
    cell = lambda v: f"{v:.{nd}f}" if isinstance(v, float) else str(v)
    head = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * frame.shape[1]
    body = ["| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, *body])

print(to_md(df))
```

```
| h | irf | se | lower90 | upper90 |
|---|---|---|---|---|
| 0 | 0.806 | 0.026 | 0.763 | 0.849 |
| 1 | 0.518 | 0.054 | 0.429 | 0.607 |
| 2 | 0.368 | 0.061 | 0.267 | 0.469 |
| 3 | 0.207 | 0.070 | 0.092 | 0.322 |
| 4 | 0.144 | 0.068 | 0.033 | 0.256 |
| 5 | 0.098 | 0.071 | -0.018 | 0.215 |
| 6 | 0.105 | 0.072 | -0.013 | 0.223 |
```

## Step 5: write it to disk

```python
from pathlib import Path
tex = df.to_latex(index=False, float_format="%.3f")
print("bytes written:", Path("irf_table.tex").write_text(tex, encoding="utf-8"))
```

```
bytes written: 364
```

Pass `encoding="utf-8"` explicitly. Python's default text encoding is
platform-dependent, and a table containing an en-dash or a Greek letter is one
`UnicodeEncodeError` away from breaking your build on someone else's machine.

## Gotchas

- **Not every result is one rectangle.** `var_irf` is `[h][response][shock]`;
  flatten with a `MultiIndex` or one frame per shock. `check_series` is a nested
  report, not a table — walk the families you want.
- **Reformat, don't round in place.** `df.round(3)` destroys precision for any
  later arithmetic; `float_format` only changes the rendering.
- **Significance stars are a policy decision.** pandas will not add them; if
  your journal wants them, generate a string column and document the thresholds
  in the table note.
- Plotting instead of tabulating? The `tsecon.results` wrappers carry
  `.plot_*()` methods that lazily import matplotlib
  (`pip install 'tsecon[plots]'`).

## See also

- Reference: [Results objects](../reference/results.md) — every bespoke wrapper
  and what it renders
- Guide: [7. Systems (VAR, Factors)](../guide/07-multivariate.md)
- Recipes: [Put confidence bands on a VAR impulse response](var-irf-bands.md) ·
  [Estimate a fiscal multiplier from an instrumented shock](fiscal-multiplier.md)
