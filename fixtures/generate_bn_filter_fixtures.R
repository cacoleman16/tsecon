# Reference leg for fixtures/bn_filters.json — the Kamber-Morley-Wong
# (2018, REStat) BN filter, run through the AUTHORS' OWN reference code.
#
# This script sources the R functions of the bnfiltering.com replication
# code ("Replication Files for Kamber, Morley & Wong ... R conversion of
# some of the MATLAB codes originally written by Ben Wong", converted by
# Luke Hartigan 2017, updated by James Morley 2022+), as packaged by
# github.com/kletts/bnfilter (R/bnf_fcns.R, commit 8af7924, 2025-05-05).
# The code itself is NOT vendored here (its DESCRIPTION carries no
# license grant); point $BNFILTER_R_DIR at a checkout instead:
#
#   git clone https://github.com/kletts/bnfilter /tmp/bnfilter
#   BNFILTER_R_DIR=/tmp/bnfilter/R Rscript fixtures/generate_bn_filter_fixtures.R ...
#
# Invoked by generate_bn_filters_fixtures.py once per stored case with
#   args: <series.csv> <p> <delta|auto> <demean: sm|nd> <out.json>
# KMW-2018 baseline options are hard-wired: delta_select = 1 (first local
# max of the amplitude-to-noise ratio, the 2018 criterion) with the
# reference grid d0 = 0.01, dt = 0.0005; ib = FALSE (unconditional-mean
# zero padding, "as in KMW2018" per the reference code's own docs); fixed
# (non-dynamic, non-outlier-adjusted) error bands.
#
# Output values are printed with %.17g so they round-trip bit-exactly.

code_dir <- Sys.getenv("BNFILTER_R_DIR")
if (!nzchar(code_dir)) {
  stop("set BNFILTER_R_DIR to the R/ directory of a github.com/kletts/bnfilter checkout")
}
source(file.path(code_dir, "bnf_fcns.R"))

args <- commandArgs(trailingOnly = TRUE)
y <- as.matrix(read.csv(args[1], header = FALSE)[[1]])
p <- as.integer(args[2])
demean <- args[4]

dy <- diff(y, lag = 1)
if (demean == "sm") {
  x <- scale(dy, center = TRUE, scale = FALSE)
} else {
  x <- dy
}

ib <- FALSE
if (args[3] == "auto") {
  delta <- select_delta(x, p, ib, delta_select = 1, d0 = 0.01, dt = 0.0005)
} else {
  delta <- as.numeric(args[3])
}

res <- BN_Filter(x, p, delta, ib)

# Fixed (non-dynamic) cycle standard error, transcribed from
# BN_Filter_stderr with dynamic_bands = FALSE, adjusted_bands = FALSE:
# sqrt(e1' Phi Sigma_X Phi' e1), Sigma_X from the vec'd Lyapunov solve,
# innovation variance from the UNPADDED AR(p)-with-constant OLS.
A <- res$aux_out$Companion
big_A <- qr.solve(eye(p^2) - (A %x% A))
vecQ <- zeros(p^2, 1)
ind_vec <- cbind(1.0, zeros(1, p - 1))
Phi <- (A %*% qr.solve(eye(p) - A))
tmp_olsvar <- olsvar(y = x, p = p, nc = FALSE)
sig2_ols_c <- tmp_olsvar$SIGMA
vecQ[1, 1] <- sig2_ols_c
Sigma_X <- matrix(big_A %*% vecQ, p, p)
cycle_se <- as.numeric(sqrt(ind_vec %*% Phi %*% Sigma_X %*% t(Phi) %*% t(ind_vec)))

amp_to_noise <- as.numeric(var(res$BN_cycle) / mean(square(res$aux_out$residuals)))

fmt <- function(v) paste0("[", paste(sprintf("%.17g", v), collapse = ","), "]")
out <- paste0(
  '{"delta":', sprintf("%.17g", delta),
  ',"cycle":', fmt(as.vector(res$BN_cycle)),
  ',"ar":', fmt(as.vector(res$aux_out$AR_coeff)),
  ',"cycle_se":', sprintf("%.17g", cycle_se),
  ',"sig2_ols_c":', sprintf("%.17g", sig2_ols_c),
  ',"amp_to_noise":', sprintf("%.17g", amp_to_noise),
  ',"dy_mean":', sprintf("%.17g", mean(dy)),
  ',"r_version":"', R.version.string, '"',
  "}"
)
writeLines(out, args[5])
