# Edge Cases & Invariants for API Proto Definitions

These are the edge cases that the protobuf schema validation layer MUST handle.
Every case listed here maps to a unit test in `internal/orchestrator/validation_test.go`.

---

## 1. ExperimentSpec Validation

### 1.1. Model Identity
| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| E1 | `model_name` is empty string `""` | Reject: `INVALID_ARGUMENT` — model name is required |
| E2 | `model_name` contains path traversal `../../etc/passwd` | Reject: `INVALID_ARGUMENT` — must be alphanumeric + hyphens only |
| E3 | `model_path` points to non-existent path | Accept at proto level; scheduler validates at runtime |
| E4 | `model_name` exceeds 256 chars | Reject: `INVALID_ARGUMENT` — name too long |

### 1.2. Tensor Parallelism Degree
| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| E5 | `tp_degrees` list is empty `[]` | Reject: must specify at least one TP degree |
| E6 | `tp_degrees` contains `0` | Reject: TP must be ≥ 1 |
| E7 | `tp_degrees` contains `3`, `5`, `6`, `7` (non-power-of-2) | Reject: TP must be power of 2 (1, 2, 4, 8) for NCCL all-reduce efficiency |
| E8 | `tp_degrees` contains `16` on an 8-GPU node | Accept at proto level; scheduler rejects if insufficient GPUs |
| E9 | `tp_degrees` contains duplicates `[2, 4, 2]` | Accept but deduplicate silently during sweep generation |
| E10 | `tp_degrees` contains negative value `-1` | Reject: must be positive integer |

### 1.3. Batch Sizes
| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| E11 | `batch_sizes` list is empty `[]` | Reject: must specify at least one batch size |
| E12 | `batch_sizes` contains `0` | Reject: batch size must be ≥ 1 |
| E13 | `batch_sizes` contains very large value `4096` | Accept: may cause OOM at runtime but is valid to test |
| E14 | `batch_sizes` contains duplicates `[1, 8, 8, 32]` | Accept but deduplicate during sweep |

### 1.4. Quantization
| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| E15 | `quantizations` is empty `[]` | Default to `[FP16]` |
| E16 | `quantizations` contains `QUANTIZATION_UNSPECIFIED` | Reject: must be a concrete quantization format |

### 1.5. Backend Selection
| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| E17 | `backend` is `BACKEND_UNSPECIFIED` | Default to `TRT_LLM` |
| E18 | `backend` is `VLLM` but TP > 1 and no `ray` config | Accept: vLLM handles multi-GPU via its own process spawning |

---

## 2. SweepDimensions — Combinatorial Explosion Guard

| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| E19 | Total sweep cells = `|TP| × |Batch| × |Quant|` exceeds 256 | Reject: sweep matrix too large; risk of multi-hour experiments |
| E20 | All dimensions have exactly 1 value (no sweep) | Accept: single-point benchmark is valid |

---

## 3. TrafficProfile Validation

| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| E21 | `concurrency` is 0 | Reject: must be ≥ 1 |
| E22 | `concurrency` is 10,000 | Reject: cap at 2048 to prevent client-side saturation |
| E23 | `duration_seconds` is 0 | Reject: must be ≥ 10 for meaningful statistics |
| E24 | `duration_seconds` exceeds 3600 (1 hour) | Warn but accept: user may want long-running soak test |
| E25 | `prompt_length` is 0 | Reject: must be ≥ 1 token |
| E26 | `prompt_length` exceeds model's max context (e.g., 131072 for Llama-3) | Accept at proto level; Triton will error at runtime |
| E27 | `max_output_tokens` is 0 | Reject: must be ≥ 1 |
| E28 | `arrival_distribution` is `DISTRIBUTION_UNSPECIFIED` | Default to `POISSON` |
| E29 | `target_qps` is set but `concurrency` is also set | Reject: mutually exclusive; use one or the other |

---

## 4. PerformanceBudget Validation (CI/CD Regression)

| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| E30 | `max_p99_ttft_ms` is 0 or negative | Reject: budget value must be > 0 |
| E31 | `max_p95_itl_ms` is 0 or negative | Reject: budget value must be > 0 |
| E32 | `min_throughput_tokens_per_sec` is 0 | Reject: must be > 0 |
| E33 | `regression_threshold_pct` is > 100 | Reject: percentage must be in (0, 100] |
| E34 | `regression_threshold_pct` is 0 | Reject: zero tolerance is unrealistic; minimum 0.1% |
| E35 | No budget specified but CI mode is enabled | Reject: CI mode requires at least one budget constraint |

---

## 5. Job Lifecycle State Machine Invariants

| # | Edge Case | Expected Behavior |
|---|-----------|-------------------|
| E36 | Transition from `COMPLETED` back to `BENCHMARKING` | Reject: terminal states are irreversible |
| E37 | Transition from `FAILED` to any state | Reject: terminal states are irreversible |
| E38 | Transition from `PENDING` directly to `BENCHMARKING` (skipping PROVISIONING) | Reject: must follow ordered state progression |
| E39 | Duplicate `StartExperiment` with same experiment ID | Reject: `ALREADY_EXISTS` error |
| E40 | `CancelExperiment` on already-completed experiment | No-op with warning, not an error |
