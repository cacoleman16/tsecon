//! The Bai–Perron dynamic program over segment SSRs.
//!
//! Given the precomputed [`SegTable`], the optimal partition of `0..T-1`
//! into `m + 1` contiguous regimes (each at least `h` observations) is
//! found by the standard break-point recursion (Bai & Perron 2003,
//! Section 4.2): `cost[r][j]` is the minimal SSR of splitting `0..=j`
//! into `r + 1` regimes, and
//!
//! ```text
//! cost[r][j] = min over t in [r*h - 1, j - h] of cost[r-1][t] + SSR(t+1, j).
//! ```
//!
//! One `O(max_breaks * T^2)` pass yields the global minimizers for every
//! `m = 1..=max_breaks` by backtracking the recorded argmins. Break dates
//! are 0-indexed as the LAST observation of each regime.

use crate::segments::SegTable;

/// Global SSR minimizers for every break count `m = 1..=max_breaks`.
pub(crate) struct GlobalPartitions {
    /// `ssr_path[m]` = minimal total SSR with `m` breaks (`ssr_path[0]` is
    /// the no-break full-sample SSR).
    pub ssr_path: Vec<f64>,
    /// `dates_by_m[m - 1]` = the `m` optimal break dates.
    pub dates_by_m: Vec<Vec<usize>>,
}

/// Run the dynamic program. Requires `(max_breaks + 1) * h <= T`, which
/// the caller has validated.
// The recursion reads and writes several rows of `cost`/`arg` through the
// textbook triple index (r, j, t); explicit indices keep it recognizably
// the Bai-Perron recursion, so the iterator rewrite the lint wants would
// hurt auditability.
#[allow(clippy::needless_range_loop)]
pub(crate) fn global_partitions(tbl: &SegTable, max_breaks: usize) -> GlobalPartitions {
    let t = tbl.t;
    let h = tbl.h;
    let mut cost = vec![vec![f64::INFINITY; t]; max_breaks + 1];
    let mut arg = vec![vec![usize::MAX; t]; max_breaks + 1];
    for j in (h - 1)..t {
        cost[0][j] = tbl.ssr(0, j);
    }
    for r in 1..=max_breaks {
        for j in ((r + 1) * h - 1)..t {
            let mut best = f64::INFINITY;
            let mut best_t = usize::MAX;
            for tt in (r * h - 1)..=(j - h) {
                let prev = cost[r - 1][tt];
                if prev.is_finite() {
                    let v = prev + tbl.ssr(tt + 1, j);
                    if v < best {
                        best = v;
                        best_t = tt;
                    }
                }
            }
            cost[r][j] = best;
            arg[r][j] = best_t;
        }
    }
    let mut ssr_path = Vec::with_capacity(max_breaks + 1);
    ssr_path.push(tbl.ssr(0, t - 1));
    let mut dates_by_m = Vec::with_capacity(max_breaks);
    for m in 1..=max_breaks {
        ssr_path.push(cost[m][t - 1]);
        let mut dates = vec![0_usize; m];
        let mut j = t - 1;
        for r in (1..=m).rev() {
            let tt = arg[r][j];
            dates[r - 1] = tt;
            j = tt;
        }
        dates_by_m.push(dates);
    }
    GlobalPartitions {
        ssr_path,
        dates_by_m,
    }
}
