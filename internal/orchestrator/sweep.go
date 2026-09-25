package orchestrator

import (
	"fmt"
	"sort"
)

// ============================================================================
// Constants
// ============================================================================

const (
	// DefaultKVCacheMemFraction is used when the user does not specify a value.
	// 85% of free GPU memory is allocated to the paged KV-cache pool.
	DefaultKVCacheMemFraction = 0.85

	// DefaultBackend is used when the user specifies BACKEND_UNSPECIFIED or empty.
	DefaultBackend = "TRT_LLM"

	// DefaultQuantization is used when the user provides an empty quantizations list.
	DefaultQuantization = "FP16"
)

// ============================================================================
// SweepCell — A Single Benchmark Configuration
// ============================================================================

// SweepCell represents one unique (TP, BatchSize, Quantization) point in the
// parameter sweep matrix. It is the atomic unit of work for the profiler.
type SweepCell struct {
	// Index is the 0-based sequential position in the execution plan.
	Index int

	// TPDegree is the tensor parallelism degree for this cell.
	TPDegree uint32

	// BatchSize is the maximum in-flight batch size for Triton's dynamic batcher.
	BatchSize uint32

	// Quantization is the weight/KV-cache precision format (e.g., "FP16", "FP8_E4M3").
	Quantization string
}

// String returns a human-readable label for logging and reporting.
func (c SweepCell) String() string {
	return fmt.Sprintf("cell[%d]{TP=%d, Batch=%d, Quant=%s}", c.Index, c.TPDegree, c.BatchSize, c.Quantization)
}

// Key returns a unique string key for deduplication and map lookups.
func (c SweepCell) Key() string {
	return fmt.Sprintf("%d:%d:%s", c.TPDegree, c.BatchSize, c.Quantization)
}

// ============================================================================
// ExecutionPlan — The Ordered List of Benchmark Cells
// ============================================================================

// ExecutionPlan is the finalized, ordered sequence of sweep cells to execute.
// It also carries the resolved defaults that will be applied to all cells.
type ExecutionPlan struct {
	// Cells is the ordered list of benchmark configurations.
	Cells []SweepCell

	// ResolvedBackend is the serving backend after default resolution.
	ResolvedBackend string

	// ResolvedKVCacheMemFraction is the KV-cache GPU memory fraction after default resolution.
	ResolvedKVCacheMemFraction float32

	// Dimensions records the deduplicated input dimensions for reporting.
	Dimensions ResolvedDimensions
}

// ResolvedDimensions captures the deduplicated and sorted dimension lists.
type ResolvedDimensions struct {
	TPDegrees     []uint32
	BatchSizes    []uint32
	Quantizations []string
}

// TotalCells returns the number of cells in the plan.
func (p *ExecutionPlan) TotalCells() int {
	return len(p.Cells)
}

// ============================================================================
// Sweep Plan Generator
// ============================================================================

// GenerateExecutionPlan transforms raw sweep input into an ordered execution plan.
//
// It performs the following steps:
//  1. Applies defaults for empty quantizations, KV-cache fraction, and backend.
//  2. Deduplicates each dimension.
//  3. Sorts dimensions (TP ascending, Batch ascending, Quant alphabetical).
//  4. Validates total cell count against MaxSweepCells.
//  5. Computes the Cartesian product in optimal execution order:
//     TP (outer) → Quantization (middle) → BatchSize (inner).
//
// Returns an error if validation fails (empty dimensions, too many cells).
func GenerateExecutionPlan(sweep SweepInput, backend string) (*ExecutionPlan, error) {
	// Step 1: Apply defaults.
	quantizations := sweep.Quantizations
	if len(quantizations) == 0 {
		quantizations = []string{DefaultQuantization}
	}

	kvFraction := sweep.KVCacheMemFraction
	if kvFraction == 0 {
		kvFraction = DefaultKVCacheMemFraction
	}

	resolvedBackend := backend
	if resolvedBackend == "" || resolvedBackend == "BACKEND_UNSPECIFIED" {
		resolvedBackend = DefaultBackend
	}

	// Step 2: Validate non-empty dimensions.
	if len(sweep.TPDegrees) == 0 {
		return nil, fmt.Errorf("sweep generation failed: tp_degrees is empty")
	}
	if len(sweep.BatchSizes) == 0 {
		return nil, fmt.Errorf("sweep generation failed: batch_sizes is empty")
	}

	// Step 3: Deduplicate.
	tpDegrees := DeduplicateUint32(sweep.TPDegrees)
	batchSizes := DeduplicateUint32(sweep.BatchSizes)
	quants := deduplicateStrings(quantizations)

	// Step 4: Sort for deterministic, optimal ordering.
	sort.Slice(tpDegrees, func(i, j int) bool { return tpDegrees[i] < tpDegrees[j] })
	sort.Slice(batchSizes, func(i, j int) bool { return batchSizes[i] < batchSizes[j] })
	sort.Strings(quants)

	// Step 5: Validate cell count.
	totalCells := len(tpDegrees) * len(batchSizes) * len(quants)
	if totalCells > MaxSweepCells {
		return nil, fmt.Errorf(
			"sweep matrix too large: %d cells (%d TP × %d Batch × %d Quant) exceeds maximum of %d",
			totalCells, len(tpDegrees), len(batchSizes), len(quants), MaxSweepCells,
		)
	}

	// Step 6: Generate Cartesian product in optimal order.
	// Outer: TP (most expensive to change — requires GPU reprovisioning).
	// Middle: Quantization (requires model weight reload).
	// Inner: BatchSize (cheapest — only reconfigures dynamic batcher).
	cells := make([]SweepCell, 0, totalCells)
	index := 0
	for _, tp := range tpDegrees {
		for _, quant := range quants {
			for _, bs := range batchSizes {
				cells = append(cells, SweepCell{
					Index:        index,
					TPDegree:     tp,
					BatchSize:    bs,
					Quantization: quant,
				})
				index++
			}
		}
	}

	return &ExecutionPlan{
		Cells:                      cells,
		ResolvedBackend:            resolvedBackend,
		ResolvedKVCacheMemFraction: kvFraction,
		Dimensions: ResolvedDimensions{
			TPDegrees:     tpDegrees,
			BatchSizes:    batchSizes,
			Quantizations: quants,
		},
	}, nil
}

// ============================================================================
// Helpers
// ============================================================================

// deduplicateStrings returns a new slice with duplicate strings removed, preserving order.
func deduplicateStrings(vals []string) []string {
	seen := make(map[string]struct{}, len(vals))
	result := make([]string, 0, len(vals))
	for _, v := range vals {
		if _, ok := seen[v]; !ok {
			seen[v] = struct{}{}
			result = append(result, v)
		}
	}
	return result
}
