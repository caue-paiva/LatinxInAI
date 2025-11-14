# KF Over Regression Weights — README

This project implements a Kalman Filter that tracks the **last layer weights** of a linear regression head on top of a small neural feature extractor. It also includes a batch Bayesian Linear Regression baseline for reference, plots for diagnostics, and guidance on how to read the results.

## Contents

* [Overview of the code](#overview-of-the-code)
* [Visualization guide](#visualization-guide)
* [How to interpret the results](#how-to-interpret-the-results)
* [Tuning tips](#tuning-tips)
* [How to run](#how-to-run)

---

## Overview of the code

### 1) Synthetic data

* Builds a 1D latent state `x_true` with drift `u` and process noise `Qx`.
* True measurement function `h_true(x)` generates noise free outputs, then Gaussian noise with variance `R` is added to form `y_meas`.
* Purpose: gives a controlled setting with known signal and noise so we can check calibration and accuracy.

Key knobs:

* `u` - drift per step of the latent state
* `Qx` - process noise variance for `x`
* `R` - measurement noise variance

### 2) Feature extractor ϕ(x)

* A small MLP is trained to predict `y_meas` from `x_true`.
* After training, we **discard the final linear layer** and keep the backbone as a **frozen feature map** ϕ with dimension `M`.
* We compute `Phi` of shape `(N, M)`, where each row is ϕ(x_k).

Why: the KF will track a **linear head** on top of these features, which makes the observation model linear in the weights.

### 3) KF over last layer weights

* **State:** weights `w_k ∈ R^M`.
* **Dynamics:** AR(1) on weights
  `w_{k+1} = ρ w_k + q_k`, with `q_k ~ N(0, Qw)`.

  * `ρ = 1.0` gives a random walk.
  * `Qw` controls plasticity.
* **Measurement:** scalar regression
  `y_k = ϕ(x_k)^T w_k + r_k`, with `r_k ~ N(0, R)`.

KF steps per time k:

* Predict
  `μ^- = ρ μ`, `P^- = ρ^2 P + Qw`
* Update
  `S = ϕ_k^T P^- ϕ_k + R`
  `K = P^- ϕ_k / S`
  `μ = μ^- + K (y_k - ϕ_k^T μ^-)`
  Joseph form for covariance:
  `P = (I - K ϕ_k^T) P^- (I - K ϕ_k^T)^T + K R K^T`

We also log:

* `y_kf_mean` - KF one step predictive mean in measurement space
* `y_kf_var` - KF one step predictive variance `S`
* `nis_hist` - normalized innovation squared `v^2 / S`
* `P_tr_hist` - trace of the weight covariance

### 4) Batch BLR baseline

* Prior on weights: `w ~ N(0, α^{-1} I)`.
* Likelihood: `y | w ~ N(Φ w, β^{-1} I)`, with `β = 1/R`.
* Posterior precision: `S_N = α I + β Φ^T Φ`
* Posterior covariance: `Σ_N = S_N^{-1}`
* Posterior mean: `m_N = β Σ_N Φ^T y`
* Predictive mean and variance at each row of `Φ`:

  * mean `= Φ m_N`
  * var `= β^{-1} + ϕ^T Σ_N ϕ`

Why: validates the KF against a closed form reference when the process is stationary. The BLR is trained on the entire datase

---

## Visualization guide

The script produces these plots:

1. **True vs measured vs BLR vs KF with 95 percent band**

   * Lines:

     * truth `y_true`
     * noisy measurement `y_meas`
     * BLR batch predictive mean
     * KF one step predictive mean
   * Shaded area:

     * KF 95 percent band from `y_kf_var` as `± 1.96 * sqrt(S)`
   * Use case:

     * Check denoising, drift tracking, and interval coverage.

2. **NIS (Normalized Innovation Squared) over time**

   * NIS `= v_k^2 / S_k` where `v_k = y_k - ϕ_k^T μ^-`.
   * Target average is near 1 when Q and R are well tuned.
   * Use case:

     * Online calibration check of the predictive variance.

3. **Trace of P over time**

   * `trace(P)` is the total weight uncertainty.
   * Use case:

     * Detect collapse or blow up of uncertainty and understand adaptivity.

You can also enable:

* **Feature or gain diagnostics** if you want to inspect directions where the filter learns the most.

---

## How to interpret the results

### RMSE

Printed like:

```
RMSE vs true: measured=..., KF=..., BLR=...
```

* Lower is better.
* `measured` vs true shows raw noise level.
* `KF` vs true shows how well the filter denoises and tracks drift.
* `BLR` vs true is a static baseline. If drift exists, KF often wins.

### NIS (Normalized Innovation Squared) mean

Printed like:

```
NIS mean (target ~1): 0.659
```

* NIS `= v^2 / S` compares **actual squared residuals** to **predicted variance**.
* If mean NIS < 1: predictive variance is **too large** on average, the filter is underconfident. Try reducing R or Qw a bit.
* If mean NIS > 1: predictive variance is **too small**, the filter is overconfident. Try increasing R or Qw.
* Goal: keep NIS near 1 over a reasonable window.

### trace(P)

Printed like:

```
Final trace(P) on weights: 15.643
```

* Sum of posterior variances of all weights.
* Rough gauge of model uncertainty in parameter space.
* If it goes to near zero: filter may be overconfident and stop adapting. Increase Qw or use ρ slightly below 1.
* If it grows without bound: filter is not stabilizing. Decrease Qw or use ρ below 1, check feature scaling.

### KF 95 percent band vs truth

* If truth falls inside the band close to 95 percent of the time, the variance is well calibrated.
* If bands are much wider than needed, NIS will be < 1.
* If bands are too narrow and truth sits outside too often, NIS will be > 1.


## How to run

1. Install dependencies

   ```
   pip install numpy matplotlib torch
   ```
2. Run the script

   ```
   python kf_over_last_layer_weights.py
   ```
3. Inspect printed metrics and the generated plots

   * Look at RMSE lines to judge point accuracy.
   * Check NIS mean for calibration.
   * Inspect trace(P) for parameter uncertainty behavior.
   * Review 95 percent bands for coverage.