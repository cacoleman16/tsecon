# Reference leg for fixtures/bn_filters.json — the Kamber-Morley-Wong
# (2018, REStat) BN filter, run through the AUTHORS' OWN reference code.
#
# This script sources the R functions of the bnfiltering.com replication
# code ("Replication Files for Kamber, Morley & Wong ... R conversion of
# some of the MATLAB codes originally written by Ben Wong", converted by
# Luke Hartigan 2017, updated by James Morley 2022+), as packaged by
# github.com/kletts/bnfilter (R/bnf_fcns.R, commit 8af7924, the repo
# HEAD when the fixture was generated). The code itself is NOT vendored
# here (its DESCRIPTION carries no license grant); point $BNFILTER_R_DIR
# at a checkout (or a directory holding bnf_fcns.R) instead:
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
# (non-dynamic, non-outlier-adjusted) error bands via the reference's own
# BN_Filter_stderr.
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

# Fixed (non-dynamic, non-outlier-adjusted) cycle standard error, from the
# reference code's own BN_Filter_stderr (window/outliers are unused on
# this path): a constant vector, one value per observation.
res_se <- BN_Filter_stderr(
  y = as.matrix(x), p = p, dynamic_bands = FALSE, ib = ib,
  window = 40, outliers = c(0), adjusted_bands = FALSE, bnf_result = res
)
cycle_se <- as.numeric(res_se$BN_cycle_se[1])
stopifnot(all(abs(res_se$BN_cycle_se - cycle_se) == 0))

# The innovation variance that SE is built from (stored for the record).
sig2_ols_c <- olsvar(y = as.matrix(x), p = p, nc = FALSE)$SIGMA

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
