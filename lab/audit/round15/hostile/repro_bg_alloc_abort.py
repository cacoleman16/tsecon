"""breusch_godfrey_test: a large-but-valid nlags allocates the T x (k + nlags) auxiliary
design with no size guard, and the process ABORTS (SIGABRT, allocator failure) instead of
raising a named refusal. Run under a 4 GB address-space cap so the machine is not exhausted.

Expected: ValueError naming nlags (as ardl_bounds_test refuses designs > 2^26 entries and
unobserved_components refuses working sets > 2^26 cells, both by name), or at least MemoryError.
Actual:   child process killed by SIGABRT (exit code -6), no Python exception.
"""
import subprocess, sys, textwrap, resource
PY = sys.executable
child = textwrap.dedent("""
    import resource, numpy as np, tsecon
    resource.setrlimit(resource.RLIMIT_AS, (4_000_000_000, 4_000_000_000))
    T, nlags = 100_000, 20_000          # valid: T > k + nlags; aux design 1e5 x 2e4 doubles = 16 GB
    rng = np.random.default_rng(0)
    y = rng.standard_normal(T); x = np.ones((T, 1))
    try:
        r = tsecon.breusch_godfrey_test(y, x, nlags=nlags)
        print("returned", r)
    except BaseException as e:
        print("raised", type(e).__name__, str(e)[:300])
""")
p = subprocess.run([PY, "-c", child], capture_output=True, text=True, timeout=120)
print("child exit code:", p.returncode, "(-6 = SIGABRT)")
print("stdout:", p.stdout.strip()[:500])
print("stderr tail:", p.stderr.strip()[-500:])
