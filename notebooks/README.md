# Notebooks

Runnable notebooks that need **no installation** — open one in Colab and it
installs `tsecon` in its first cell.

| Notebook | What it covers | |
|---|---|---|
| **[1 · tsecon in five minutes](01_tour.ipynb)** | Screen a series, fit a VAR, read an impulse response, put bands on it, and see what the library refuses to guess for you. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cacoleman16/tsecon/blob/main/notebooks/01_tour.ipynb) |
| **[2 · IRF bands, and LP vs VAR](02_irf_bands_and_lp_vs_var.ipynb)** | Delta-method vs bootstrap bands, a coverage check that shows where they fail, and a measured answer to the local-projections-versus-VAR question. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cacoleman16/tsecon/blob/main/notebooks/02_irf_bands_and_lp_vs_var.ipynb) |
| **[3 · Replicating Blanchard-Quah (1989)](03_blanchard_quah.ipynb)** | Long-run restrictions end to end on real data: identification, convergence of the restriction, sign normalisation, the four-panel figure, and an FEVD. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cacoleman16/tsecon/blob/main/notebooks/03_blanchard_quah.ipynb) |

## How these are maintained

Each notebook is generated from a plain Python source file (`*_src.py`) in the
percent-cell format. The source is what gets edited and what CI executes:

```sh
python notebooks/01_tour_src.py     # runs exactly what the notebook contains
python notebooks/build.py           # regenerate the .ipynb files
```

Keeping the source as real Python means the notebooks are diffable and
testable, and can never drift into claiming output the code does not produce.
The committed `.ipynb` files carry **no outputs** — the reader generates them.

If you change a `_src.py`, re-run `build.py` and commit both.
