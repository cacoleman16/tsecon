# Reference run for fixtures/lpdid.json — LP-DiD (Dube, Girardi, Jordà &
# Taylor 2025, J. Applied Econometrics 40(7), doi:10.1002/jae.70000)
# estimated through fixest, the engine of the authors' own reference code.
#
# Invoked by generate_lpdid_fixtures.py as
#
#     Rscript fixtures/generate_lpdid_fixtures.R <panelA.csv> <panelB.csv> <out.csv>
#
# This script is a line-faithful transcription of the authors' example
# implementations (github.com/danielegirardi/lpdid, fetched 2026-08-19):
#
#   * absorbing VW event study + pooled: LP_DiD_R_example_VW.R lines
#     136-178 (post filter `D_treat == 1 | F{h}_treat == 0`, pre filter
#     `D_treat == 1 | treat == 0`, `feols(D{h}y ~ D_treat | time,
#     vcov = ~unit)`, pooled post clean through t+H);
#   * the equally-weighted (reweighted) estimator: LP_DiD_R_example_EW.R
#     lines 138-166 (`get_reweights`: residual of D_treat on time FEs
#     within the clean sample, normalized over switchers, controls
#     inheriting their cell's switcher weight, regression weight = the
#     inverse) and lines 177-198/267-284 (pre horizons and the pooled pre
#     use the h=0 weights; the pooled post uses the h=H weights);
#     the same construction ships in the R port's get_weights
#     (github.com/alexCardazzi/lpdid, R/func.R lines 45-66);
#   * non-absorbing treatment with a stabilization window L:
#     LPDiD_nonabsorbing_example.do lines 187-212 (the CCS_h / CCS_mh
#     clean-control indicators, variant 5b) — including Stata's missing
#     semantics: a status change outside the observed panel counts as
#     clean (`abs(L#.D.treat) != 1` is true for missing lags);
#   * never-treated-only controls: the Stata lpdid package's
#     `nevertreated` option — the control-side condition becomes "unit is
#     never treated in the observed sample".
#
# The clustered-SE small-sample convention is the authors' own choice,
# `setFixest_ssc(ssc(adj = TRUE, cluster.adj = TRUE))` (their
# LP_DiD_R_example_VW.R line 18, "Match reghdfe small-sample correction"):
# (n-1)/(n-K) * G/(G-1) with K counting the slope plus all absorbed
# period-effect levels (verified against fixest 0.14.2 at machine
# precision during fixture generation).

suppressMessages(library(fixest))
setFixest_ssc(ssc(adj = TRUE, cluster.adj = TRUE))

args <- commandArgs(trailingOnly = TRUE)
panelA_csv <- args[1]
panelB_csv <- args[2]
out_csv <- args[3]

lagm <- function(m, k) {
  if (k == 0) return(m)
  cbind(matrix(NA_real_, nrow(m), k), m[, seq_len(ncol(m) - k), drop = FALSE])
}
leadm <- function(m, k) {
  if (k == 0) return(m)
  cbind(m[, (k + 1):ncol(m), drop = FALSE], matrix(NA_real_, nrow(m), k))
}

read_panel <- function(path) {
  df <- read.csv(path)
  df <- df[order(df$unit, df$time), ]
  N <- length(unique(df$unit))
  T <- length(unique(df$time))
  list(
    y = matrix(df$y, nrow = N, byrow = TRUE),
    d = matrix(df$d, nrow = N, byrow = TRUE),
    N = N, T = T
  )
}

results <- data.frame()
emit <- function(case, horizon, m, dat) {
  results <<- rbind(results, data.frame(
    case = case, horizon = horizon,
    coef = unname(coef(m)["D_treat"]), se = unname(se(m)["D_treat"]),
    nobs = nobs(m), nsw = sum(dat$D_treat == 1)
  ))
}

# One LP-DiD case. mode = "abs" or "nonabs"; L = stabilization window.
run_case <- function(case, p, Q, H, mode, L = 0, rw = FALSE, nt = FALSE,
                     pooled = FALSE) {
  N <- p$N; T <- p$T; y <- p$y; d <- p$d
  unit <- rep(seq_len(N), times = T)
  time <- rep(seq_len(T), each = N)
  Dtr <- d - lagm(d, 1)
  LY <- lagm(y, 1)
  dt <- as.vector(Dtr)
  dvec <- as.vector(d)
  ever <- rep(apply(d, 1, max) == 1, times = T)

  if (mode == "nonabs") {
    # CCS indicators per LPDiD_nonabsorbing_example.do lines 191-212;
    # missing lags/leads count as clean (Stata `abs(.) != 1` is true).
    chg <- abs(Dtr) == 1
    chg[is.na(chg)] <- FALSE
    C0 <- cbind(0, t(apply(chg, 1, cumsum))) # C0[, t+1] = changes at 1..t
    changes_in <- function(a, b) {
      # matrix of change counts in time window [t+a, t+b] per (unit, t)
      out <- matrix(0, N, T)
      for (t in seq_len(T)) {
        lo <- max(t + a, 1); hi <- min(t + b, T)
        if (lo <= hi) out[, t] <- C0[, hi + 1] - C0[, lo]
      }
      out
    }
    no_lag <- function(width) as.vector(changes_in(-width, -1) == 0)
    no_lead <- function(h) as.vector(changes_in(1, h) == 0)
  }

  clean_post <- function(h) {
    if (mode == "abs") {
      if (nt) ctrl <- !ever
      else {
        fh <- as.vector(leadm(d, h))
        ctrl <- !is.na(fh) & fh == 0
      }
      !is.na(dt) & (dt == 1 | (dt == 0 & ctrl))
    } else {
      base <- !is.na(dt) & dt >= 0 & no_lag(L) & no_lead(h)
      if (nt) base & (dt == 1 | !ever) else base
    }
  }
  clean_pre <- function(j) {
    if (mode == "abs") {
      ctrl <- if (nt) !ever else dvec == 0
      !is.na(dt) & (dt == 1 | (dt == 0 & ctrl))
    } else {
      base <- !is.na(dt) & dt >= 0 & no_lag(L + j - 1)
      if (nt) base & (dt == 1 | !ever) else base
    }
  }

  # get_reweights transcription (LP_DiD_R_example_EW.R lines 138-166):
  # built on the clean sample at horizon h (no outcome filter — matching
  # the reference; only the normalization differs on outcome-incomplete
  # cells, and WLS is invariant to weight scale).
  weights_map <- function(h) {
    keep <- clean_post(h)
    dat <- data.frame(unit = unit[keep], time = time[keep], D_treat = dt[keep])
    mw <- feols(D_treat ~ 1 | time, data = dat, vcov = "iid")
    num <- as.numeric(residuals(mw))
    num[dat$D_treat != 1] <- NA
    den <- sum(num, na.rm = TRUE)
    w <- num / den
    gw <- ave(w, dat$time, FUN = function(x) suppressWarnings(max(x, na.rm = TRUE)))
    gw[is.infinite(gw)] <- NA
    w[is.na(w)] <- gw[is.na(w)]
    dat$reweight <- 1 / w
    dat[, c("unit", "time", "reweight")]
  }

  est <- function(dy, keep, wmap) {
    dat <- data.frame(unit = unit, time = time, Dy = dy, D_treat = dt)[keep, ]
    dat <- dat[!is.na(dat$Dy), ]
    if (!is.null(wmap)) {
      key <- paste(dat$unit, dat$time)
      dat$reweight <- wmap$reweight[match(key, paste(wmap$unit, wmap$time))]
      dat <- dat[!is.na(dat$reweight), ]
      stopifnot(all(is.finite(dat$reweight)), all(dat$reweight > 0))
      m <- feols(Dy ~ D_treat | time, data = dat, weights = ~reweight, vcov = ~unit)
    } else {
      m <- feols(Dy ~ D_treat | time, data = dat, vcov = ~unit)
    }
    list(m = m, dat = dat)
  }

  w0 <- if (rw) weights_map(0) else NULL
  for (h in 0:H) {
    wmap <- if (rw) weights_map(h) else NULL
    r <- est(as.vector(leadm(y, h) - LY), clean_post(h), wmap)
    emit(case, as.character(h), r$m, r$dat)
  }
  if (Q >= 2) {
    for (j in 2:Q) {
      r <- est(as.vector(lagm(y, j) - LY), clean_pre(j), w0)
      emit(case, as.character(-j), r$m, r$dat)
    }
  }
  if (pooled) {
    mp <- Reduce(`+`, lapply(0:H, function(k) leadm(y, k))) / (H + 1)
    wH <- if (rw) weights_map(H) else NULL
    r <- est(as.vector(mp - LY), clean_post(H), wH)
    emit(case, "pooled_post", r$m, r$dat)
    if (Q >= 2) {
      pp <- Reduce(`+`, lapply(2:Q, function(k) lagm(y, k))) / (Q - 1)
      r <- est(as.vector(pp - LY), clean_pre(Q), w0)
      emit(case, "pooled_pre", r$m, r$dat)
    }
  }
}

pa <- read_panel(panelA_csv)
pb <- read_panel(panelB_csv)

run_case("A_vw", pa, Q = 4, H = 6, mode = "abs", pooled = TRUE)
run_case("A_rw", pa, Q = 4, H = 6, mode = "abs", rw = TRUE, pooled = TRUE)
run_case("A_nt", pa, Q = 4, H = 6, mode = "abs", nt = TRUE, pooled = TRUE)
run_case("B_vw", pb, Q = 3, H = 4, mode = "nonabs", L = 3, pooled = TRUE)
run_case("B_rw", pb, Q = 3, H = 4, mode = "nonabs", L = 3, rw = TRUE)
run_case("B_nt", pb, Q = 3, H = 4, mode = "nonabs", L = 3, nt = TRUE)

write.csv(results, out_csv, row.names = FALSE)
cat(sprintf("wrote %s (%d rows; fixest %s, R %s)\n", out_csv, nrow(results),
            as.character(packageVersion("fixest")), R.version.string))
