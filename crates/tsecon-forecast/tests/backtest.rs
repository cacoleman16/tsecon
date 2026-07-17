//! Property and guardrail tests for the pseudo-out-of-sample backtesting
//! engine: the pinned `(origin, horizon, target)` alignment, expanding vs
//! rolling training windows, the infrequent-refit contract, and the
//! degenerate-input errors.

use std::cell::RefCell;

use tsecon_forecast::{mae, mse, rmse, Backtest, BacktestResult, ForecastError, Window};

/// Strictly increasing pseudo-random series: monotone (so every value is
/// unique, letting a perfect-foresight closure recover the origin from the
/// last training value) yet irregular enough to make the invariants
/// non-trivial.
fn increasing_series(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut y = Vec::with_capacity(n);
    let mut level = 100.0;
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        level += 0.2 + u; // strictly positive increment => strictly increasing
        y.push(level);
    }
    y
}

/// A perfect-foresight forecaster: it captures the whole series and returns
/// the true future values, recovering the origin `t` from the last training
/// observation (unique because the series is strictly increasing).
fn perfect_foresight(
    y: &[f64],
) -> impl FnMut(&[f64], usize) -> Result<Vec<f64>, ForecastError> + '_ {
    move |train: &[f64], max_h: usize| {
        let last = *train.last().expect("non-empty training window");
        let t = y.iter().position(|&v| v == last).expect("origin locatable");
        Ok((1..=max_h).map(|h| y[t + h]).collect())
    }
}

#[test]
fn perfect_foresight_gives_zero_errors_and_correct_alignment() {
    let y = increasing_series(60, 7);
    for window in [
        Window::Expanding { min_train: 10 },
        Window::Rolling { width: 8 },
    ] {
        for refit_every in [1usize, 3, 5] {
            let bt = Backtest::new(window, 4, refit_every).unwrap();
            let res = bt.run(&y, perfect_foresight(&y)).unwrap();
            // Alignment: target at row i, horizon h is exactly y[origin + h].
            for h in 1..=res.horizon() {
                let targets = res.targets(h).unwrap();
                let errors = res.errors(h).unwrap();
                for (i, &t) in res.origins().iter().enumerate() {
                    assert_eq!(targets[i], y[t + h], "target index alignment");
                    assert!(
                        errors[i].abs() < 1e-12,
                        "perfect foresight error should be ~0 (window {window:?}, \
                         refit {refit_every}, h={h}, origin {t}): {}",
                        errors[i]
                    );
                }
            }
        }
    }
}

#[test]
fn expanding_grows_and_rolling_is_fixed_width() {
    let y = increasing_series(50, 11);

    // Record the training-slice length at each refit call.
    let lengths = RefCell::new(Vec::<usize>::new());
    let bt = Backtest::new(Window::Expanding { min_train: 12 }, 3, 1).unwrap();
    bt.run(&y, |train: &[f64], max_h: usize| {
        lengths.borrow_mut().push(train.len());
        Ok(vec![*train.last().unwrap(); max_h])
    })
    .unwrap();
    // refit_every = 1 => one call per origin; expanding length = min_train + i.
    let exp = lengths.borrow().clone();
    for (i, &len) in exp.iter().enumerate() {
        assert_eq!(
            len,
            12 + i,
            "expanding training window grows by one per origin"
        );
    }

    let rlengths = RefCell::new(Vec::<usize>::new());
    let bt = Backtest::new(Window::Rolling { width: 9 }, 3, 1).unwrap();
    bt.run(&y, |train: &[f64], max_h: usize| {
        rlengths.borrow_mut().push(train.len());
        Ok(vec![*train.last().unwrap(); max_h])
    })
    .unwrap();
    for &len in rlengths.borrow().iter() {
        assert_eq!(len, 9, "rolling training window is fixed width");
    }
}

/// A naive (random-walk) forecaster: every horizon is the last training value.
fn naive_forecaster(train: &[f64], max_h: usize) -> Result<Vec<f64>, ForecastError> {
    Ok(vec![*train.last().unwrap(); max_h])
}

#[test]
fn naive_backtest_reproduces_hand_rolled_slicing() {
    let y = increasing_series(45, 3);
    let (min_train, h) = (10usize, 3usize);
    let bt = Backtest::new(Window::Expanding { min_train }, h, 1).unwrap();
    let res = bt.run(&y, naive_forecaster).unwrap();

    // Hand-rolled origins and expectations.
    let n = y.len();
    let t0 = min_train - 1;
    let last_origin = n - 1 - h;
    let expected_origins: Vec<usize> = (t0..=last_origin).collect();
    assert_eq!(res.origins(), expected_origins.as_slice());
    assert_eq!(res.n_origins(), expected_origins.len());

    for hh in 1..=h {
        let forecasts = res.forecasts(hh).unwrap();
        let targets = res.targets(hh).unwrap();
        let errors = res.errors(hh).unwrap();
        for (i, &t) in expected_origins.iter().enumerate() {
            assert_eq!(
                forecasts[i], y[t],
                "naive forecast is the last training value"
            );
            assert_eq!(targets[i], y[t + hh], "target is y[t+h]");
            assert_eq!(errors[i], y[t + hh] - y[t], "error is actual - forecast");
        }
    }

    // Same for a rolling window.
    let width = 8usize;
    let bt = Backtest::new(Window::Rolling { width }, h, 1).unwrap();
    let res = bt.run(&y, naive_forecaster).unwrap();
    let t0 = width - 1;
    for (i, t) in (t0..=last_origin).enumerate() {
        assert_eq!(res.forecasts(1).unwrap()[i], y[t]);
        assert_eq!(res.targets(2).unwrap()[i], y[t + 2]);
    }
}

#[test]
fn refit_every_freezes_the_fit_between_refits() {
    // With refit_every = k, a naive forecaster fit at refit origin t_r is
    // reused for the next k-1 origins: the horizon-1 forecast at origin t_r+s
    // is y[t_r] (frozen), not y[t_r+s].
    let y = increasing_series(40, 21);
    let (min_train, h, k) = (10usize, 2usize, 4usize);
    let bt = Backtest::new(Window::Expanding { min_train }, h, k).unwrap();
    let res = bt.run(&y, naive_forecaster).unwrap();

    let t0 = min_train - 1;
    let f1 = res.forecasts(1).unwrap();
    for (i, &t) in res.origins().iter().enumerate() {
        // The refit origin governing this origin is the block start.
        let block_index = i / k;
        let t_r = t0 + block_index * k;
        assert_eq!(t, t0 + i, "origins are contiguous");
        assert_eq!(
            f1[i], y[t_r],
            "origin {t} reuses the naive forecast frozen at refit origin {t_r}"
        );
    }
    // A refit at every origin (k = 1) would instead give y[t]; confirm they
    // differ somewhere (the series is strictly increasing).
    let bt1 = Backtest::new(Window::Expanding { min_train }, h, 1).unwrap();
    let res1 = bt1.run(&y, naive_forecaster).unwrap();
    assert_ne!(res1.forecasts(1).unwrap(), f1);
}

#[test]
fn accuracy_table_matches_direct_measures() {
    let y = increasing_series(50, 9);
    let bt = Backtest::new(Window::Expanding { min_train: 12 }, 3, 1).unwrap();
    let res: BacktestResult = bt.run(&y, naive_forecaster).unwrap();
    let table = res.accuracy_table(1).unwrap();
    assert_eq!(table.len(), 3);
    for (h, row) in (1..=3).zip(table.iter()) {
        assert_eq!(row.name, format!("h={h}"));
        let targets = res.targets(h).unwrap();
        let forecasts = res.forecasts(h).unwrap();
        assert_eq!(row.mse, mse(targets, forecasts).unwrap());
        assert_eq!(row.rmse, rmse(targets, forecasts).unwrap());
        assert_eq!(row.mae, mae(targets, forecasts).unwrap());
        assert!(row.mase.is_some() && row.rmsse.is_some());
    }
}

#[test]
fn backtest_guardrails_teach_on_degenerate_inputs() {
    let y = increasing_series(30, 1);

    // Invalid scheme parameters.
    assert!(matches!(
        Backtest::new(Window::Expanding { min_train: 0 }, 1, 1).unwrap_err(),
        ForecastError::InvalidBacktestParam { .. }
    ));
    assert!(matches!(
        Backtest::new(Window::Rolling { width: 0 }, 1, 1).unwrap_err(),
        ForecastError::InvalidBacktestParam { .. }
    ));
    assert!(matches!(
        Backtest::new(Window::Expanding { min_train: 5 }, 0, 1).unwrap_err(),
        ForecastError::InvalidBacktestParam { .. }
    ));
    assert!(matches!(
        Backtest::new(Window::Expanding { min_train: 5 }, 1, 0).unwrap_err(),
        ForecastError::InvalidBacktestParam { .. }
    ));

    // No origins: training window + horizon exhaust the series.
    let bt = Backtest::new(Window::Expanding { min_train: 28 }, 5, 1).unwrap();
    assert!(matches!(
        bt.run(&y, naive_forecaster).unwrap_err(),
        ForecastError::NoBacktestOrigins { .. }
    ));

    // Forecaster returns the wrong number of forecasts.
    let bt = Backtest::new(Window::Expanding { min_train: 10 }, 3, 1).unwrap();
    let err = bt
        .run(&y, |_train: &[f64], _max_h: usize| Ok(vec![0.0])) // should be 3
        .unwrap_err();
    assert!(matches!(
        err,
        ForecastError::ForecasterOutputLen {
            expected: 3,
            actual: 1,
            ..
        }
    ));
    assert!(format!("{err}").contains("forecaster closure returned"));

    // Non-finite series.
    let mut bad = y.clone();
    bad[3] = f64::NAN;
    assert!(matches!(
        bt.run(&bad, naive_forecaster).unwrap_err(),
        ForecastError::NonFinite { index: 3, .. }
    ));

    // A closure error propagates unchanged.
    let bt = Backtest::new(Window::Expanding { min_train: 10 }, 3, 1).unwrap();
    let err = bt
        .run(&y, |_t: &[f64], _h: usize| {
            Err(ForecastError::EmptyComparison)
        })
        .unwrap_err();
    assert_eq!(err, ForecastError::EmptyComparison);

    // Horizon out of range on the result.
    let res = bt.run(&y, naive_forecaster).unwrap();
    assert!(matches!(
        res.errors(0).unwrap_err(),
        ForecastError::HorizonOutOfRange { h: 0, max_h: 3 }
    ));
    assert!(matches!(
        res.forecasts(4).unwrap_err(),
        ForecastError::HorizonOutOfRange { h: 4, max_h: 3 }
    ));
}
