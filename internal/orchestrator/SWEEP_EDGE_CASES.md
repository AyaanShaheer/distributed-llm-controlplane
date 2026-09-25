# Edge Cases & Invariants: Sweep Matrix Generator

The sweep engine transforms raw user input dimensions into an ordered
execution plan of benchmark cells. Each cell is a unique (TP, BatchSize, Quantization)
configuration to test on the GPU cluster.

---

## Execution Order Strategy

Changing TP degree requires reprovisioning GPU allocations (expensive: ~30–120s).
Changing quantization may require reloading model weights (~10–30s).
Changing batch size only reconfigures the Triton dynamic batcher (~1s).

**Optimal order: TP (outer) → Quantization (middle) → BatchSize (inner)**

This minimizes the number of expensive reprovisioning operations.

---

## Sweep Generation Edge Cases (SW-*)

| # | Test Case | Input | Expected Result |
|---|-----------|-------|-----------------|
| SW1 | Minimal single-point sweep | TP=[1], Batch=[32], Quant=[FP16] | 1 cell |
| SW2 | Standard 2×3×1 grid | TP=[1,2], Batch=[1,8,32], Quant=[FP16] | 6 cells, ordered TP→Quant→Batch |
| SW3 | Full 4×4×2 grid | TP=[1,2,4,8], Batch=[1,8,32,128], Quant=[FP16,FP8_E4M3] | 32 cells |
| SW4 | Empty quantizations defaults to [FP16] | TP=[1], Batch=[1], Quant=[] | 1 cell with FP16 |
| SW5 | Duplicate TP degrees are deduplicated | TP=[2,4,2,4], Batch=[1], Quant=[FP16] | 2 cells (TP=2, TP=4) |
| SW6 | Duplicate batch sizes are deduplicated | TP=[1], Batch=[8,8,8], Quant=[FP16] | 1 cell |
| SW7 | Duplicate quantizations are deduplicated | TP=[1], Batch=[1], Quant=[FP16,FP16,BF16] | 2 cells |
| SW8 | TP degrees are sorted ascending | TP=[8,2,4,1], Batch=[1], Quant=[FP16] | Order: TP=1, TP=2, TP=4, TP=8 |
| SW9 | Batch sizes are sorted ascending | TP=[1], Batch=[128,1,32,8], Quant=[FP16] | Order: 1, 8, 32, 128 |
| SW10 | Maximum cells exactly at limit (256) | 4×8×8 = 256 | Accept: exactly 256 cells |
| SW11 | Exceeds maximum cells | 4×10×7 = 280 | Reject: too many cells |
| SW12 | KV cache fraction defaults to 0.85 | Fraction=0.0 | Plan uses 0.85 |
| SW13 | KV cache fraction preserves user value | Fraction=0.92 | Plan uses 0.92 |
| SW14 | Empty TP degrees | TP=[], Batch=[1], Quant=[FP16] | Reject: validation error |
| SW15 | Empty batch sizes | TP=[1], Batch=[], Quant=[FP16] | Reject: validation error |
| SW16 | Backend defaults from UNSPECIFIED to TRT_LLM | Backend="" | Plan uses "TRT_LLM" |

---

## Execution Plan Properties

| # | Property | Invariant |
|---|----------|-----------|
| P1 | Cell uniqueness | No two cells in the plan have the same (TP, Batch, Quant) tuple |
| P2 | TP ordering | Cells are grouped by TP degree in ascending order |
| P3 | Within-TP ordering | Within the same TP group, cells are sub-sorted by Quant then BatchSize |
| P4 | Cell count | len(cells) == |dedup(TP)| × |dedup(Batch)| × |dedup(Quant)| |
| P5 | Cell indices | Each cell has a 0-based sequential index |
