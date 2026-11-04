# credit_var api benchmark (WS_CREDIT_VAR_API S6, C.32)

## Environment

| Item | Value |
|---|---|
| Command | `C:\Users\mharvey\Documents\code\riskquant\.venv\Scripts\python.exe scripts\solutions\credit_var\bench_credit_var_api.py` |
| Working directory | `C:\Users\mharvey\Documents\code\riskquant` |
| Started (UTC) | `2026-11-04T14:34:29+00:00` |
| CPU model | `13th Gen Intel(R) Core(TM) i7-1355U` |
| Logical processors (os.cpu_count) | `12` |
| RAM | `31.7 GiB (34017017856 bytes)` |
| OS | `Windows-10-10.0.26200-SP0` |
| Python | `3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]` |
| Python executable | `C:\Users\mharvey\Documents\code\riskquant\.venv\Scripts\python.exe` |
| riskquant commit (git, tracked files) | `184a9e848010ac57278c4f8e5edfabf9de624c65 dirty=false` |
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
| Work directory | `C:\Users\mharvey\Documents\code\riskquant\.bench_credit_var\20261104T143429Z` |
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
| empty | ok | 7.7978 / 8.0271 / 8.1355 | 7.7978, 7.8028, 8.0409, 8.0271, 8.1355 | 0->11, 0->11, 0->11, 0->11, 0->11 |
| warm | ok | 1.6978 / 1.7221 / 1.7808 | 1.6978, 1.7221, 1.7117, 1.7482, 1.7808 | 11->11, 11->11, 11->11, 11->11, 11->11 |

## M2: first and second run_credit_var call in one process (N = 100, S = 250000)

| Model | Numba cache | Status | First call s | Second call s | First - second s (median) | .nbi before -> after |
|---|---|---|---|---|---|---|
| forced_default | empty | ok | 0.4996 / 0.5344 / 0.6108 | 0.4241 / 0.4430 / 0.4763 | 0.1000 | 0->11, 0->11, 0->11, 0->11, 0->11 |
| forced_default | warm | ok | 0.5357 / 0.5635 / 0.5822 | 0.4519 / 0.4566 / 0.4913 | 0.1029 | 11->11, 11->11, 11->11, 11->11, 11->11 |
| sf_gaussian | empty | ok | 1.0638 / 1.2249 / 1.3639 | 1.1453 / 1.2168 / 1.2587 | 0.0081 | 0->11, 0->11, 0->11, 0->11, 0->11 |
| sf_gaussian | warm | ok | 1.0722 / 1.1298 / 1.1866 | 1.1075 / 1.1187 / 1.1876 | 0.0111 | 11->11, 11->11, 11->11, 11->11, 11->11 |
| mf_students_t | empty | ok | 1.3646 / 1.3960 / 1.4907 | 1.3110 / 1.3346 / 1.3468 | 0.0615 | 0->11, 0->11, 0->11, 0->11, 0->11 |
| mf_students_t | warm | ok | 1.3564 / 1.4174 / 1.4915 | 1.3631 / 1.3716 / 1.4397 | 0.0458 | 11->11, 11->11, 11->11, 11->11, 11->11 |

## M3: run_credit_var, S = 250000, warm Numba cache

| Model | N | Status | Wall s | Peak RSS MiB | Peak RSS before call MiB (median) | Payload sha256[:16] |
|---|---|---|---|---|---|---|
| forced_default | 10 | ok | 0.0741 / 0.0909 / 0.0977 | 284.9 / 285.1 / 285.2 | 206.3 | 03b07b8f97a9273a |
| forced_default | 100 | ok | 0.4877 / 0.5299 / 0.5647 | 971.5 / 971.6 / 972.2 | 206.3 | 9470d28b593ee662 |
| forced_default | 1000 | ok | 4.2084 / 4.2626 / 4.5512 | 4266.7 / 4267.3 / 4267.5 | 206.5 | 4403bc8f961305e5 |
| sf_gaussian | 10 | ok | 0.1521 / 0.1964 / 0.2009 | 284.5 / 285.0 / 285.3 | 206.6 | 8b354b74ae400fce |
| sf_gaussian | 100 | ok | 1.0885 / 1.1505 / 1.2198 | 971.4 / 971.8 / 972.0 | 206.7 | 8ac4f217d80cbba8 |
| sf_gaussian | 1000 | ok | 8.9520 / 9.4025 / 9.5862 | 5934.2 / 5934.6 / 5935.4 | 206.9 | 161dc2e2c1d15d23 |
| mf_students_t | 10 | ok | 0.4608 / 0.5070 / 0.5200 | 285.1 / 285.6 / 285.6 | 207.0 | ffb41d68f36a3d29 |
| mf_students_t | 100 | ok | 1.2662 / 1.3172 / 1.4019 | 971.7 / 972.7 / 972.9 | 207.0 | 8b3edd570d84e124 |
| mf_students_t | 1000 | ok | 9.8966 / 10.7197 / 11.2962 | 5944.3 / 5945.2 / 5945.5 | 209.6 | 233a0d5d8052e39d |

## M4: simulate_portfolio_losses (default max_samples), S = 250000, warm Numba cache

| Model | N | Status | Wall s | Peak RSS MiB | Peak RSS before call MiB (median) | Payload sha256[:16] |
|---|---|---|---|---|---|---|
| forced_default | 10 | ok | 0.0432 / 0.0570 / 0.0589 | 245.7 / 245.9 / 246.0 | 205.2 | 03b07b8f97a9273a |
| forced_default | 100 | ok | 0.1686 / 0.1715 / 0.2162 | 589.2 / 589.7 / 589.7 | 205.5 | 9470d28b593ee662 |
| forced_default | 1000 | ok | 1.3374 / 1.3779 / 1.4636 | 4023.9 / 4024.5 / 4025.1 | 206.2 | 4403bc8f961305e5 |
| sf_gaussian | 10 | ok | 0.0771 / 0.0921 / 0.0998 | 264.2 / 264.5 / 265.0 | 205.1 | 8b354b74ae400fce |
| sf_gaussian | 100 | ok | 0.5241 / 0.5565 / 0.5857 | 779.6 / 779.9 / 780.3 | 205.4 | 8ac4f217d80cbba8 |
| sf_gaussian | 1000 | ok | 5.2451 / 5.7977 / 6.2775 | 5931.4 / 5931.8 / 5932.0 | 206.2 | 161dc2e2c1d15d23 |
| mf_students_t | 10 | ok | 0.4059 / 0.4248 / 0.5547 | 270.6 / 270.9 / 271.1 | 205.7 | ffb41d68f36a3d29 |
| mf_students_t | 100 | ok | 0.7927 / 0.8022 / 0.8784 | 786.4 / 786.5 / 787.0 | 205.9 | 8b3edd570d84e124 |
| mf_students_t | 1000 | ok | 5.8277 / 5.9684 / 6.2571 | 5942.2 / 5942.5 / 5942.7 | 209.5 | 233a0d5d8052e39d |

## M5: first engine_version computation in a process

Callable: `riskquant.solutions.credit_var.provenance:engine_version`

| Status | First call s | Second call s | Values |
|---|---|---|---|
| ok | 0.110719 / 0.141275 / 0.153669 | 0.000002 / 0.000003 / 0.000003 | `0.2.0+src.14c4a0f4cb4983fd92ef2da14b0dc9bb4312dce546371c20f4fef681062c8f6e` |

## Failures

none

