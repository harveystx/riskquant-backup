# credit_var api benchmark (WS_CREDIT_VAR_API S6, C.32)

## Environment

| Item | Value |
|---|---|
| Command | `C:\Users\mharvey\Documents\code\riskquant\.venv\Scripts\python.exe scripts\solutions\credit_var\bench_credit_var_api.py` |
| Working directory | `C:\Users\mharvey\Documents\code\riskquant` |
| Started (UTC) | `2026-11-03T16:37:04+00:00` |
| CPU model | `13th Gen Intel(R) Core(TM) i7-1355U` |
| Logical processors (os.cpu_count) | `12` |
| RAM | `31.7 GiB (34017017856 bytes)` |
| OS | `Windows-10-10.0.26200-SP0` |
| Python | `3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]` |
| Python executable | `C:\Users\mharvey\Documents\code\riskquant\.venv\Scripts\python.exe` |
| riskquant commit (git, tracked files) | `e8d5d8d3526e07b3a4e866308d58a3aa78b53d4a dirty=false` |
| riskquant package directory | `C:\Users\mharvey\Documents\code\riskquant\src\riskquant` |
| NUMBA_NUM_THREADS (environment) | `unset` |
| Thread variables in the parent environment | `none` |
| NumPy | `2.4.6` |
| SciPy | `1.17.1` |
| Numba | `0.66.0` |
| pydantic | `2.13.4` |
| numba.threading_layer() | `omp` |
| numba.config.NUMBA_NUM_THREADS | `12` |
| numba.get_num_threads() | `12` |
| riskquant module file | `C:\Users\mharvey\Documents\code\riskquant\src\riskquant\__init__.py` |
| riskquant distribution version | `0.2.0` |
| Work directory | `C:\Users\mharvey\Documents\code\riskquant\.bench_credit_var\20261103T163704Z` |
| Threading layers seen in measured children | `omp` |

## Method

- Each case runs in a fresh child process, 5 repetitions. Cells give min / median / max. Wall time is `time.perf_counter` inside the child.
- Book: obligor_uid 1..N; PD ~ U[0.005, 0.05); EAD ~ lognormal(mu = ln(1e6), sigma = 1.0); LGD 0.45; generator `numpy.random.default_rng(20261030)`.
- Request: alphas (0.95, 0.975, 0.99, 0.999); seed 12345; sf_gaussian params {rho: 0.15}; mf_students_t params {nu: 7.0} and loadings 0.3 on SYS_01, SYS_02, SYS_03 for each obligor; forced_default params {}.
- Numba cache: `NUMBA_CACHE_DIR` per child. Empty: one fresh directory per repetition. Warm: one shared directory, filled by the environment probe (import, then one small call per model of each entry point). `.nbi` columns count cache index files before and after the child.
- M3 and M4: each child first calls the same entry point with the same model at N = 10, S = 1000 (warm-up), then times the full call. Peak RSS is the process peak after the timed call; the pre-call column is the process peak after import, warm-up and request construction.
- Peak RSS: Windows `PeakWorkingSetSize` (GetProcessMemoryInfo); POSIX `ru_maxrss`. 1 MiB = 2**20 bytes.
- Request validation (`CreditVarRequest.model_validate`) is outside every timed region. A call-time validation inside the entry point is inside it.
- M1 child: `python -P -c <program>`; it loads only `sys` and `time` before the timed import. The Python bytecode cache (`__pycache__`) is not controlled.
- After S1, the first `run_credit_var` call of a process also computes `engine_version` (M5). M2 includes that cost; M3 and M4 do not.

## M1: import riskquant.solutions.credit_var.api

| Numba cache | Status | Wall s (min / median / max) | All repetitions s | .nbi before -> after |
|---|---|---|---|---|
| empty | ok | 7.8093 / 8.0532 / 8.2795 | 7.8093, 7.8952, 8.0532, 8.1382, 8.2795 | 0->11, 0->11, 0->11, 0->11, 0->11 |
| warm | ok | 1.7205 / 1.7755 / 1.8121 | 1.7205, 1.7926, 1.8121, 1.7755, 1.7701 | 11->11, 11->11, 11->11, 11->11, 11->11 |

## M2: first and second run_credit_var call in one process (N = 100, S = 250000)

| Model | Numba cache | Status | First call s | Second call s | First - second s (median) | .nbi before -> after |
|---|---|---|---|---|---|---|
| forced_default | empty | ok | 0.5658 / 0.5700 / 0.5964 | 0.4367 / 0.4583 / 0.5003 | 0.1116 | 0->11, 0->11, 0->11, 0->11, 0->11 |
| forced_default | warm | ok | 0.5392 / 0.5688 / 0.6536 | 0.4566 / 0.4655 / 0.4765 | 0.1026 | 11->11, 11->11, 11->11, 11->11, 11->11 |
| sf_gaussian | empty | ok | 1.0957 / 1.1140 / 1.2507 | 1.0445 / 1.0635 / 1.0839 | 0.0696 | 0->11, 0->11, 0->11, 0->11, 0->11 |
| sf_gaussian | warm | ok | 1.0408 / 1.1034 / 1.1257 | 1.0146 / 1.0407 / 1.0678 | 0.0590 | 11->11, 11->11, 11->11, 11->11, 11->11 |
| mf_students_t | empty | ok | 1.2860 / 1.2885 / 1.3477 | 1.1799 / 1.2589 / 1.3514 | 0.0617 | 0->11, 0->11, 0->11, 0->11, 0->11 |
| mf_students_t | warm | ok | 1.2574 / 1.2934 / 1.3109 | 1.1771 / 1.3069 / 1.3776 | -0.0195 | 11->11, 11->11, 11->11, 11->11, 11->11 |

## M3: run_credit_var, S = 250000, warm Numba cache

| Model | N | Status | Wall s | Peak RSS MiB | Peak RSS before call MiB (median) | Payload sha256[:16] |
|---|---|---|---|---|---|---|
| forced_default | 10 | ok | 0.0855 / 0.0892 / 0.1264 | 284.8 / 285.0 / 285.4 | 206.3 | 03b07b8f97a9273a |
| forced_default | 100 | ok | 0.4961 / 0.5551 / 0.5770 | 971.7 / 972.0 / 972.2 | 206.4 | 9470d28b593ee662 |
| forced_default | 1000 | ok | 4.4481 / 4.6949 / 5.6011 | 4266.6 / 4267.1 / 4267.5 | 206.5 | 4403bc8f961305e5 |
| sf_gaussian | 10 | ok | 0.1439 / 0.1648 / 0.1870 | 284.3 / 284.9 / 285.1 | 206.4 | 8b354b74ae400fce |
| sf_gaussian | 100 | ok | 0.9896 / 1.0470 / 1.0710 | 971.3 / 971.6 / 972.3 | 206.4 | 8ac4f217d80cbba8 |
| sf_gaussian | 1000 | ok | 8.1753 / 8.3748 / 8.4255 | 5934.1 / 5934.5 / 5934.6 | 206.8 | 161dc2e2c1d15d23 |
| mf_students_t | 10 | ok | 0.4616 / 0.4788 / 0.4989 | 285.3 / 285.4 / 285.6 | 206.7 | ffb41d68f36a3d29 |
| mf_students_t | 100 | ok | 1.2358 / 1.3181 / 1.3700 | 972.2 / 972.4 / 972.8 | 206.9 | 8b3edd570d84e124 |
| mf_students_t | 1000 | ok | 9.3889 / 9.4542 / 9.6596 | 5944.9 / 5945.2 / 5945.4 | 210.4 | 233a0d5d8052e39d |

## M4: simulate_portfolio_losses (default max_samples), S = 250000, warm Numba cache

| Model | N | Status | Wall s | Peak RSS MiB | Peak RSS before call MiB (median) | Payload sha256[:16] |
|---|---|---|---|---|---|---|
| forced_default | 10 | ok | 0.0438 / 0.0581 / 0.0627 | 245.4 / 245.6 / 246.0 | 204.9 | 03b07b8f97a9273a |
| forced_default | 100 | ok | 0.1748 / 0.2127 / 0.2376 | 588.9 / 589.5 / 589.9 | 205.4 | 9470d28b593ee662 |
| forced_default | 1000 | ok | 1.4159 / 1.4277 / 1.4627 | 4024.4 / 4024.4 / 4024.7 | 206.2 | 4403bc8f961305e5 |
| sf_gaussian | 10 | ok | 0.0784 / 0.0869 / 0.1084 | 264.3 / 264.5 / 264.8 | 205.1 | 8b354b74ae400fce |
| sf_gaussian | 100 | ok | 0.5286 / 0.5601 / 0.5888 | 779.2 / 779.9 / 780.0 | 205.3 | 8ac4f217d80cbba8 |
| sf_gaussian | 1000 | ok | 5.2396 / 5.4668 / 5.5048 | 5931.8 / 5932.0 / 5932.4 | 206.3 | 161dc2e2c1d15d23 |
| mf_students_t | 10 | ok | 0.4101 / 0.4168 / 0.4267 | 270.5 / 271.3 / 271.6 | 206.0 | ffb41d68f36a3d29 |
| mf_students_t | 100 | ok | 0.8183 / 0.8433 / 0.8881 | 786.3 / 786.4 / 787.2 | 206.1 | 8b3edd570d84e124 |
| mf_students_t | 1000 | ok | 6.0872 / 6.1994 / 6.2742 | 5942.1 / 5942.8 / 5943.0 | 209.9 | 233a0d5d8052e39d |

## M5: first engine_version computation in a process

Callable: `riskquant.solutions.credit_var.provenance:engine_version`

| Status | First call s | Second call s | Values |
|---|---|---|---|
| ok | 0.127272 / 0.141136 / 0.166536 | 0.000002 / 0.000003 / 0.000003 | `0.2.0+src.cea3a36f4736d2d5a41b5c3f5d0f6d1f5bb60eb4139a60c7240c9f9d2c9e0201` |

## Failures

none

