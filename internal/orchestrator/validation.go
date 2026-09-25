// Package validation provides server-side validation for all incoming API messages.
//
// Protobuf does not enforce business invariants (e.g., "TP must be a power of 2").
// This package bridges that gap with deterministic, pure-function validators that
// return structured error lists without any I/O or side effects.
package validation

import (
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"
)

// ============================================================================
// Constants & Constraints
// ============================================================================

const (
	// MaxModelNameLen is the maximum allowed length for a model name.
	MaxModelNameLen = 256

	// MaxSweepCells is the maximum total cells in a sweep matrix to prevent
	// combinatorial explosion (|TP| × |Batch| × |Quant|).
	MaxSweepCells = 256

	// MaxConcurrency caps the traffic generator's concurrent streams to prevent
	// client-side CPU saturation from skewing TTFT measurements.
	MaxConcurrency = 2048

	// MinBenchmarkDuration is the minimum benchmark window for statistical significance.
	MinBenchmarkDuration = 10 * time.Second

	// MaxBenchmarkDuration is the upper warning threshold (not a hard reject).
	MaxBenchmarkDuration = 1 * time.Hour

	// MaxRegressionThresholdPct is the upper bound for regression tolerance.
	MaxRegressionThresholdPct = 100.0

	// MinRegressionThresholdPct prevents zero-tolerance budgets which are unrealistic.
	MinRegressionThresholdPct = 0.1
)

// modelNamePattern enforces safe, portable model identifiers.
// Allows alphanumeric, hyphens, underscores, and dots. Must start with alphanumeric.
var modelNamePattern = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._-]*$`)

// ============================================================================
// Error Types
// ============================================================================

// FieldError represents a single validation violation on a specific field.
type FieldError struct {
	Field   string // Dot-delimited field path (e.g., "spec.sweep.tp_degrees").
	Message string // Human-readable description of the violation.
}

func (e FieldError) Error() string {
	return fmt.Sprintf("%s: %s", e.Field, e.Message)
}

// ValidationResult collects all validation errors found in a single pass.
// A nil or empty Errors slice indicates a valid input.
type ValidationResult struct {
	Errors   []FieldError
	Warnings []FieldError
}

// IsValid returns true if no errors were recorded.
func (v *ValidationResult) IsValid() bool {
	return len(v.Errors) == 0
}

// HasWarnings returns true if warnings were recorded.
func (v *ValidationResult) HasWarnings() bool {
	return len(v.Warnings) > 0
}

// addError appends a validation error.
func (v *ValidationResult) addError(field, msg string) {
	v.Errors = append(v.Errors, FieldError{Field: field, Message: msg})
}

// addWarning appends a validation warning (non-blocking).
func (v *ValidationResult) addWarning(field, msg string) {
	v.Warnings = append(v.Warnings, FieldError{Field: field, Message: msg})
}

// ErrorMessages returns all error messages as a single newline-delimited string.
func (v *ValidationResult) ErrorMessages() string {
	msgs := make([]string, len(v.Errors))
	for i, e := range v.Errors {
		msgs[i] = e.Error()
	}
	return strings.Join(msgs, "\n")
}

// ============================================================================
// Input Structs (Proto-independent for testability)
// ============================================================================

// These structs mirror the protobuf messages but are plain Go types.
// This decouples validation logic from protobuf generated code, making
// tests runnable without `protoc` compilation.

// ModelSpecInput mirrors ModelSpec from the proto definition.
type ModelSpecInput struct {
	ModelName       string
	ModelPath       string
	MaxSeqLen       uint32
	NumLayers       uint32
	NumKVHeads      uint32
	HeadDim         uint32
	TotalParameters uint64
}

// SweepInput mirrors SweepDimensions.
type SweepInput struct {
	TPDegrees         []uint32
	BatchSizes        []uint32
	Quantizations     []string // String enum names: "FP16", "BF16", "FP8_E4M3", etc.
	KVCacheMemFraction float32
}

// TrafficInput mirrors TrafficProfile.
type TrafficInput struct {
	Concurrency         uint32
	DurationSeconds     int64 // Converted from protobuf Duration.
	PromptLength        uint32
	MaxOutputTokens     uint32
	ArrivalDistribution string // String enum name.
	TargetQPS           float32
	WarmupRequests      uint32
}

// BudgetInput mirrors PerformanceBudget.
type BudgetInput struct {
	MaxP99TTFT_MS         float64
	MaxP95ITL_MS          float64
	MinThroughputTPS      float64
	RegressionThresholdPct float64
	BaselineExperimentID  string
}

// ExperimentInput mirrors ExperimentSpec.
type ExperimentInput struct {
	ExperimentID string
	Model        ModelSpecInput
	Sweep        SweepInput
	Backend      string // String enum name.
	Traffic      TrafficInput
	Budget       *BudgetInput // nil if no budget specified.
	Labels       map[string]string
	CIMode       bool
}

// ============================================================================
// Validators
// ============================================================================

// ValidateExperimentSpec performs comprehensive validation of an experiment submission.
// It returns all violations found in a single pass (does not fail-fast).
func ValidateExperimentSpec(input ExperimentInput) ValidationResult {
	var result ValidationResult

	validateModelSpec(input.Model, &result)
	validateSweepDimensions(input.Sweep, &result)
	validateTrafficProfile(input.Traffic, &result)
	validateBackend(input.Backend, &result)

	if input.CIMode && input.Budget == nil {
		result.addError("spec.ci_mode", "CI mode enabled but no performance budget specified")
	}
	if input.Budget != nil {
		validateBudget(*input.Budget, &result)
	}

	return result
}

// validateModelSpec checks the model identity fields.
func validateModelSpec(m ModelSpecInput, result *ValidationResult) {
	// E1: Empty model name.
	if m.ModelName == "" {
		result.addError("spec.model.model_name", "model name is required")
		return // Skip further model name checks if empty.
	}

	// E4: Name too long.
	if len(m.ModelName) > MaxModelNameLen {
		result.addError("spec.model.model_name",
			fmt.Sprintf("model name exceeds maximum length of %d characters", MaxModelNameLen))
	}

	// E2: Path traversal or invalid characters.
	if !modelNamePattern.MatchString(m.ModelName) {
		result.addError("spec.model.model_name",
			"model name must match ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ (alphanumeric, hyphens, dots, underscores)")
	}

	// Architectural metadata validation.
	if m.MaxSeqLen == 0 {
		result.addError("spec.model.max_seq_len", "max sequence length must be > 0")
	}
	if m.NumLayers == 0 {
		result.addError("spec.model.num_layers", "number of layers must be > 0")
	}
	if m.NumKVHeads == 0 {
		result.addError("spec.model.num_kv_heads", "number of KV heads must be > 0")
	}
	if m.HeadDim == 0 {
		result.addError("spec.model.head_dim", "head dimension must be > 0")
	}
}

// validateSweepDimensions checks the parameter sweep configuration.
func validateSweepDimensions(s SweepInput, result *ValidationResult) {
	// E5: Empty TP degrees.
	if len(s.TPDegrees) == 0 {
		result.addError("spec.sweep.tp_degrees", "must specify at least one tensor parallelism degree")
	}

	for i, tp := range s.TPDegrees {
		// E10: Negative value (impossible for uint32, but 0 check).
		// E6: Zero value.
		if tp == 0 {
			result.addError("spec.sweep.tp_degrees",
				fmt.Sprintf("tp_degrees[%d]: must be ≥ 1", i))
			continue
		}
		// E7: Non-power-of-2.
		if !isPowerOfTwo(tp) {
			result.addError("spec.sweep.tp_degrees",
				fmt.Sprintf("tp_degrees[%d]=%d: must be a power of 2 (1, 2, 4, 8) for NCCL all-reduce efficiency", i, tp))
		}
	}

	// E11: Empty batch sizes.
	if len(s.BatchSizes) == 0 {
		result.addError("spec.sweep.batch_sizes", "must specify at least one batch size")
	}

	for i, bs := range s.BatchSizes {
		// E12: Zero batch size.
		if bs == 0 {
			result.addError("spec.sweep.batch_sizes",
				fmt.Sprintf("batch_sizes[%d]: must be ≥ 1", i))
		}
	}

	// Default quantizations to FP16 if empty (E15).
	quantCount := len(s.Quantizations)
	if quantCount == 0 {
		quantCount = 1 // Will default to FP16.
	}

	// E16: Reject UNSPECIFIED quantization.
	for i, q := range s.Quantizations {
		if q == "QUANTIZATION_UNSPECIFIED" || q == "" {
			result.addError("spec.sweep.quantizations",
				fmt.Sprintf("quantizations[%d]: must be a concrete quantization format, not UNSPECIFIED", i))
		}
	}

	// E19: Combinatorial explosion guard.
	tpCount := len(s.TPDegrees)
	bsCount := len(s.BatchSizes)
	if tpCount == 0 {
		tpCount = 1
	}
	if bsCount == 0 {
		bsCount = 1
	}
	totalCells := tpCount * bsCount * quantCount
	if totalCells > MaxSweepCells {
		result.addError("spec.sweep",
			fmt.Sprintf("total sweep cells (%d = %d×%d×%d) exceeds maximum of %d",
				totalCells, tpCount, bsCount, quantCount, MaxSweepCells))
	}

	// KV cache fraction bounds check.
	if s.KVCacheMemFraction != 0 {
		if s.KVCacheMemFraction < 0.1 || s.KVCacheMemFraction > 0.99 {
			result.addError("spec.sweep.kv_cache_mem_fraction",
				"must be between 0.1 and 0.99 (e.g., 0.85 for 85% of free GPU memory)")
		}
	}
}

// validateTrafficProfile checks the load injection parameters.
func validateTrafficProfile(t TrafficInput, result *ValidationResult) {
	// E29: Mutual exclusion between concurrency and target_qps.
	if t.Concurrency > 0 && t.TargetQPS > 0 {
		result.addError("spec.traffic",
			"concurrency and target_qps are mutually exclusive; set one to 0")
		return
	}

	if t.Concurrency == 0 && t.TargetQPS == 0 {
		result.addError("spec.traffic",
			"either concurrency or target_qps must be specified (both are 0)")
	}

	// E21, E22: Concurrency bounds.
	if t.Concurrency > MaxConcurrency {
		result.addError("spec.traffic.concurrency",
			fmt.Sprintf("concurrency %d exceeds maximum of %d to prevent client-side saturation", t.Concurrency, MaxConcurrency))
	}

	// E23: Duration too short.
	dur := time.Duration(t.DurationSeconds) * time.Second
	if dur < MinBenchmarkDuration {
		result.addError("spec.traffic.duration",
			fmt.Sprintf("benchmark duration %v is below minimum of %v for statistical significance", dur, MinBenchmarkDuration))
	}

	// E24: Duration very long (warning, not error).
	if dur > MaxBenchmarkDuration {
		result.addWarning("spec.traffic.duration",
			fmt.Sprintf("benchmark duration %v exceeds %v; this may be a long-running soak test", dur, MaxBenchmarkDuration))
	}

	// E25: Zero prompt length.
	if t.PromptLength == 0 {
		result.addError("spec.traffic.prompt_length", "prompt length must be ≥ 1 token")
	}

	// E27: Zero output tokens.
	if t.MaxOutputTokens == 0 {
		result.addError("spec.traffic.max_output_tokens", "max output tokens must be ≥ 1")
	}

	// E28: Unspecified arrival distribution (handled as default, not error).
	// The defaulting happens in the experiment engine, not validation.
}

// validateBackend checks the serving backend selection.
func validateBackend(backend string, result *ValidationResult) {
	validBackends := map[string]bool{
		"BACKEND_UNSPECIFIED": true, // Defaults to TRT_LLM.
		"TRT_LLM":            true,
		"VLLM":               true,
	}
	if backend != "" && !validBackends[backend] {
		result.addError("spec.backend",
			fmt.Sprintf("unknown backend %q; must be one of: TRT_LLM, VLLM", backend))
	}
}

// validateBudget checks the CI/CD performance regression budget.
func validateBudget(b BudgetInput, result *ValidationResult) {
	// E30: TTFT budget.
	if b.MaxP99TTFT_MS <= 0 {
		result.addError("spec.budget.max_p99_ttft_ms", "must be > 0")
	}

	// E31: ITL budget.
	if b.MaxP95ITL_MS <= 0 {
		result.addError("spec.budget.max_p95_itl_ms", "must be > 0")
	}

	// E32: Throughput floor.
	if b.MinThroughputTPS <= 0 {
		result.addError("spec.budget.min_throughput_tokens_per_sec", "must be > 0")
	}

	// E33, E34: Regression threshold.
	if b.RegressionThresholdPct <= 0 || b.RegressionThresholdPct > MaxRegressionThresholdPct {
		result.addError("spec.budget.regression_threshold_pct",
			fmt.Sprintf("must be in (%.1f, %.1f]", MinRegressionThresholdPct, MaxRegressionThresholdPct))
	} else if b.RegressionThresholdPct < MinRegressionThresholdPct {
		result.addError("spec.budget.regression_threshold_pct",
			fmt.Sprintf("%.2f%% is below minimum tolerance of %.1f%%", b.RegressionThresholdPct, MinRegressionThresholdPct))
	}
}

// ============================================================================
// Helpers
// ============================================================================

// isPowerOfTwo checks if n > 0 and n is a power of 2.
func isPowerOfTwo(n uint32) bool {
	return n > 0 && (n&(n-1)) == 0
}

// DeduplicateUint32 returns a new slice with duplicates removed, preserving order.
func DeduplicateUint32(vals []uint32) []uint32 {
	seen := make(map[uint32]struct{}, len(vals))
	result := make([]uint32, 0, len(vals))
	for _, v := range vals {
		if _, ok := seen[v]; !ok {
			seen[v] = struct{}{}
			result = append(result, v)
		}
	}
	return result
}

// EstimateKVCacheBytesPerToken calculates the KV cache memory per token.
// Formula: 2 × numLayers × numKVHeads × headDim × bytesPerElement
func EstimateKVCacheBytesPerToken(numLayers, numKVHeads, headDim uint32, bytesPerElement uint32) uint64 {
	return 2 * uint64(numLayers) * uint64(numKVHeads) * uint64(headDim) * uint64(bytesPerElement)
}

// EstimateWeightMemoryBytes calculates the static model weight memory in bytes.
func EstimateWeightMemoryBytes(totalParams uint64, bytesPerParam float64) uint64 {
	return uint64(math.Ceil(float64(totalParams) * bytesPerParam))
}
