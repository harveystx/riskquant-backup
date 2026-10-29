# WS_CREDIT_VAR_API / A_ORACLES: independent oracles per catalog key

Path: docs/workstreams/WS_CREDIT_VAR_API/A_ORACLES.md
Revision: 1
Used by: S4
Snapshot: SNAPSHOT commit=c060f06 dirty=false content_hash=fa836966c1293f33

## 1. Rules

1. Each test calls only `riskquant.solutions.credit_var.api` (C.27). The oracle
   uses numpy, scipy and closed forms that the test writes out. The oracle
   calls no riskquant function.
2. Use EAD = 1 and LGD = 1 per obligor unless a key states other values. Then
   the portfolio loss L is an integer (a count, or a weighted count with integer
   weights), and every loss atom is an exact float.
3. Population definitions. F is the CDF of L. The atoms are l_0 < l_1 < ...
   ```text
   VaR_a = min{ l : F(l) >= a }
   ES_a  = ( E[L 1{L > VaR_a}] + VaR_a (F(VaR_a) - a) ) / (1 - a)
   EL    = E[L]
   ```
4. Atom-gap rule. S = n_scenarios, d = 6 sqrt(a (1 - a) / S). Use a level a only
   if F(l_prev) <= a - d and F(VaR_a) >= a + d, where l_prev is the atom below
   VaR_a. The test asserts this condition on the oracle before the engine runs.
   Under the rule, the engine VaR equals the oracle VaR whatever order-statistic
   index the engine uses (F-30).
5. Assertions:
   ```text
   VaR: engine == oracle (exact)
   EL:  |EL_mc - EL| <= 6 sd(L) / sqrt(S)
   ES:  |ES_mc - ES| <= 6 s_ES / sqrt((1 - a) S)
        s_ES^2 = ( E[L^2 1{L > VaR}] + VaR^2 (F(VaR) - a) ) / (1 - a) - ES^2
   ```
6. Oracle self-checks: the pmf sums to 1 within 1e-10. Where a closed-form mean
   exists, the oracle EL equals it within 1e-8 (for every copula key: N x PD).
7. Seed 20261029. S = 100_000. Candidate levels: 0.95, 0.975, 0.99, 0.995,
   0.999. Use each candidate that passes rule 4. The test fails if fewer than 2
   pass; then change the book, not the rule.
8. EL_mc is `summary.loss_mean`. VaR and ES come from `summary.at(a)`.

## 2. Oracle per key

K1 `forced_default` (S4.T1)
: Book: 4 obligors, EAD (1e6, 2.5e6, 3e5, 7e5), LGD (0.45, 0.6, 1.0, 0.25).
  Oracle: L = sum(EAD x LGD) in every scenario. Assert VaR = ES = loss_mean = L
  at every level (rel 1e-12), loss_std_dev = 0, each es_contribution =
  EAD_i x LGD_i (rel 1e-12), expected_loss_by_model["FORCED_DEFAULT"] = L.
  Rules 4 and 5 do not apply.

K2 `independent_bernoulli` (S4.T2)
: Book: 3 obligors, EAD (1, 2, 4), PD (0.05, 0.10, 0.20). Oracle: enumerate the
  8 default vectors; L = sum(EAD_i D_i); P = prod of PD_i or (1 - PD_i).

K3 `beta_binomial` (S4.T3)
: Book: N = 20, alpha = 2, beta = 38, master PD = 0.30 (ignored by contract).
  Oracle: K ~ BetaBinomial(20, 2, 38) (`scipy.stats.betabinom`); L = K.

K4 `beta_binomial_grouped` (S4.T4)
: Pool A: 10 obligors, (alpha, beta) = (1, 99), EAD 1. Pool B: 10 obligors,
  (4, 16), EAD 2. Oracle: L = K_A + 2 K_B with K_A and K_B independent
  BetaBinomial counts; pmf by convolution of pmf_A with pmf_B placed on even
  atoms.

K5 `stress_regime_bernoulli` (S4.T5)
: Book: N = 20, PD = 0.02, stress_prob q = 0.10, stress_pd_multiplier m = 3.
  Oracle: K ~ (1 - q) Bin(20, 0.02) + q Bin(20, min(1, 0.06)).

K6 `stress_regime_bernoulli_grouped` (S4.T6)
: Group A: 10 obligors, PD 0.02, q 0.10, m 3, EAD 1. Group B: 10 obligors,
  PD 0.05, q 0.20, m 4, EAD 2. Oracle: independent regimes; L = K_A + 2 K_B;
  pmf by convolution of the two K5-type mixtures.

K7 `sf_gaussian` (S4.T7)
: Book: N = 20, PD = 0.02, rho = 0.15.
  ```text
  p(z)     = Phi( (Phi^-1(PD) - sqrt(rho) z) / sqrt(1 - rho) )
  P(K = k) = integral Bin(k; N, p(z)) phi(z) dz
  ```
  Quadrature: probabilists' Gauss-Hermite, 200 nodes
  (`numpy.polynomial.hermite_e.hermegauss`, weights divided by sqrt(2 pi)).

K8 `sf_gaussian_basel` (S4.T8)
: K7 with rho = R(PD), written out in the test (Basel IRB corporate weighting):
  ```text
  R(PD) = 0.12 (1 - e^(-50 PD)) / (1 - e^(-50))
        + 0.24 (1 - (1 - e^(-50 PD)) / (1 - e^(-50)))
  ```
  The test does not call `basel_corporate_correlation`.

K9 `sf_students_t` (S4.T9)
: Book: N = 20, PD = 0.02, rho = 0.15, nu = 7. W ~ chi2(nu), shared.
  ```text
  p(z, w)  = Phi( (t_nu^-1(PD) sqrt(w / nu) - sqrt(rho) z) / sqrt(1 - rho) )
  P(K = k) = double integral Bin(k; N, p(z, w)) phi(z) f_chi2(w; nu) dz dw
  ```
  Quadrature: Gauss-Hermite 120 nodes in z; generalized Gauss-Laguerre 120
  nodes in u = w / 2 with exponent nu/2 - 1 (`scipy.special.roots_genlaguerre`),
  weights divided by Gamma(nu/2). VERIFY FIRST in the kernel: default when
  T_i <= t_nu^-1(PD) with T_i = sqrt(nu / W) A_i.

K10 `sf_students_t_basel` (S4.T10)
: K9 with rho = R(PD) from K8.

K11 `mf_gaussian` (S4.T11)
: Factors F1, F2. Sector A: 10 obligors, loading F1 = 0.5. Sector B: 10
  obligors, loadings F1 = 0.3, F2 = 0.4. PD = 0.02 for all; EAD 1.
  ```text
  p_A(z1)     = Phi( (Phi^-1(PD) - 0.5 z1) / sqrt(0.75) )
  p_B(z1, z2) = Phi( (Phi^-1(PD) - 0.3 z1 - 0.4 z2) / sqrt(0.75) )
  P(L = l)    = double integral [Bin(.; 10, p_A) * Bin(.; 10, p_B)](l) phi(z1) phi(z2)
  ```
  `*` is discrete convolution. Quadrature: Gauss-Hermite 80 x 80.

K12 `mf_students_t` (S4.T12)
: K11 with nu = 7 and one shared W ~ chi2(nu): each threshold Phi^-1(PD) becomes
  t_nu^-1(PD) sqrt(w / nu). Quadrature: 60 x 60 Gauss-Hermite x 60
  generalized Gauss-Laguerre.

K13 `marshall_olkin` (S4.T13)
: Book: N = 20, PD = 0.05, systemic_pd s = 0.02.
  ```text
  p_idio   = 1 - (1 - PD) / (1 - s)
  P(K = k) = s 1{k = N} + (1 - s) Bin(k; N, p_idio)
  ```
  Self-check: EL = N x PD.

## 3. Coverage limits

- The oracles cover EL, VaR and ES. Attribution fields are covered for
  forced_default only.
- Each copula key uses one PD value and a homogeneous book or two homogeneous
  sectors.
- The basel keys test R(PD) at one PD value.
- The oracles do not measure the Monte Carlo convergence rate.
