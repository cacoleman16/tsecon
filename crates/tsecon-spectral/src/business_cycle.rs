//! Reading a spectrum in the macroeconomist's units: periods, not
//! frequencies.
//!
//! A spectral estimate is indexed by frequency `f` in cycles per unit time,
//! but business-cycle analysis speaks in *periods* — "fluctuations of 6 to
//! 32 quarters". The two are reciprocal, `period = 1 / f`, and this module
//! makes that mapping explicit so a user can ask which frequencies carry the
//! variance of a given band.

use crate::periodogram::PowerSpectrum;

/// Convert a frequency (cycles per unit time) to its period (time units per
/// cycle): `period = 1 / freq`.
///
/// The zero frequency (a permanent level / DC component) maps to an infinite
/// period; this returns `f64::INFINITY` there rather than dividing by zero.
pub fn frequency_to_period(freq: f64) -> f64 {
    if freq == 0.0 {
        f64::INFINITY
    } else {
        1.0 / freq
    }
}

/// Convert a period (time units per cycle) to its frequency (cycles per unit
/// time): `freq = 1 / period`. An infinite period maps to frequency `0`.
pub fn period_to_frequency(period: f64) -> f64 {
    if period.is_infinite() {
        0.0
    } else {
        1.0 / period
    }
}

/// A band of oscillation lengths expressed in the series' own time units.
///
/// Because period and frequency are reciprocal, a period band
/// `[shortest, longest]` is the frequency band `[1/longest, 1/shortest]`.
/// The canonical macro example is the Baxter-King (1999) business-cycle
/// window of 6 to 32 quarters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodBand {
    /// Shortest oscillation retained (smallest period, highest frequency).
    pub shortest: f64,
    /// Longest oscillation retained (largest period, lowest frequency).
    pub longest: f64,
}

impl PeriodBand {
    /// Build a band from a `[shortest, longest]` period range (time units).
    ///
    /// # Panics
    /// Never panics; callers that pass `shortest > longest` simply get an
    /// empty band (nothing satisfies the frequency bounds).
    pub fn new(shortest: f64, longest: f64) -> Self {
        PeriodBand { shortest, longest }
    }

    /// The classic business-cycle band for quarterly data: 6 to 32 quarters
    /// (Baxter-King 1999; Burns-Mitchell 1946 defined the cycle as 1.5-8
    /// years, i.e. 6-32 quarters).
    pub fn business_cycle_quarterly() -> Self {
        PeriodBand::new(6.0, 32.0)
    }

    /// The business-cycle band for monthly data: 18 to 96 months (the same
    /// 1.5-8 year window).
    pub fn business_cycle_monthly() -> Self {
        PeriodBand::new(18.0, 96.0)
    }

    /// Lower frequency edge of the band, `1 / longest`.
    pub fn low_frequency(&self) -> f64 {
        period_to_frequency(self.longest)
    }

    /// Upper frequency edge of the band, `1 / shortest`.
    pub fn high_frequency(&self) -> f64 {
        period_to_frequency(self.shortest)
    }

    /// Whether `freq` (cycles per unit time) falls in this band, i.e. its
    /// period `1/freq` lies in `[shortest, longest]`. Endpoints are
    /// inclusive.
    pub fn contains_frequency(&self, freq: f64) -> bool {
        freq >= self.low_frequency() && freq <= self.high_frequency()
    }

    /// Share of a one-sided PSD's total mass carried by frequencies in this
    /// band: `sum(psd[i] : freq[i] in band) / sum(psd)`.
    ///
    /// With density scaling this is the fraction of the series' variance
    /// attributable to the band — the number a macro user quotes as "the
    /// business cycle explains X% of the variance". Returns `0.0` for a
    /// spectrum with zero total mass.
    pub fn variance_share(&self, spectrum: &PowerSpectrum) -> f64 {
        let total: f64 = spectrum.psd.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        let in_band: f64 = spectrum
            .freqs
            .iter()
            .zip(spectrum.psd.iter())
            .filter(|(f, _)| self.contains_frequency(**f))
            .map(|(_, p)| *p)
            .sum();
        in_band / total
    }
}
