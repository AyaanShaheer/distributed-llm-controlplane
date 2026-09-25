package validation

import (
	"fmt"
	"strings"
	"testing"
)

// ============================================================================
// Test Helpers
// ============================================================================

// validModelSpec returns a known-good ModelSpecInput for composing test cases.
func validModelSpec() ModelSpecInput {
	return ModelSpecInput{
		ModelName:       "llama-3-8b-instruct",
		ModelPath:       "/models/llama-3-8b",
		MaxSeqLen:       8192,
		NumLayers:       32,
		NumKVHeads:      8,
		HeadDim:         128,
		TotalParameters: 8_030_000_000,
	}
}

// validSweep returns a known-good SweepInput.
func validSweep() SweepInput {
	return SweepInput{
		TPDegrees:         []uint32{1, 2},
		BatchSizes:        []uint32{1, 8, 32},
		Quantizations:     []string{"FP16"},
		KVCacheMemFraction: 0.85,
	}
}

// validTraffic returns a known-good TrafficInput.
func validTraffic() TrafficInput {
	return TrafficInput{
		Concurrency:         32,
		DurationSeconds:     60,
		PromptLength:        512,
		MaxOutputTokens:     128,
		ArrivalDistribution: "POISSON",
		TargetQPS:           0, // Closed-loop mode (concurrency-based).
		WarmupRequests:       10,
	}
}

// validBudget returns a known-good BudgetInput.
func validBudget() BudgetInput {
	return BudgetInput{
		MaxP99TTFT_MS:          120.0,
		MaxP95ITL_MS:           22.0,
		MinThroughputTPS:       1800.0,
		RegressionThresholdPct: 5.0,
	}
}

// validExperiment returns a fully valid ExperimentInput.
func validExperiment() ExperimentInput {
	return ExperimentInput{
		Model:   validModelSpec(),
		Sweep:   validSweep(),
		Backend: "TRT_LLM",
		Traffic: validTraffic(),
		Budget:  nil,
		CIMode:  false,
	}
}

// requireValid asserts that the result has no errors.
func requireValid(t *testing.T, result ValidationResult) {
	t.Helper()
	if !result.IsValid() {
		t.Fatalf("expected valid, got %d errors:\n%s", len(result.Errors), result.ErrorMessages())
	}
}

// requireErrors asserts at least one error matches the given field prefix.
func requireErrors(t *testing.T, result ValidationResult, fieldSubstring string) {
	t.Helper()
	if result.IsValid() {
		t.Fatal("expected validation errors, got none")
	}
	for _, e := range result.Errors {
		if strings.Contains(e.Field, fieldSubstring) {
			return // Found a matching error.
		}
	}
	t.Errorf("expected error on field containing %q, but errors were:\n%s",
		fieldSubstring, result.ErrorMessages())
}

// requireWarnings asserts at least one warning matches the given field prefix.
func requireWarnings(t *testing.T, result ValidationResult, fieldSubstring string) {
	t.Helper()
	if !result.HasWarnings() {
		t.Fatal("expected warnings, got none")
	}
	for _, w := range result.Warnings {
		if strings.Contains(w.Field, fieldSubstring) {
			return
		}
	}
	t.Errorf("expected warning on field containing %q, but warnings were: %+v",
		fieldSubstring, result.Warnings)
}

// ============================================================================
// Baseline Sanity Test
// ============================================================================

func TestValidExperiment_PassesValidation(t *testing.T) {
	exp := validExperiment()
	result := ValidateExperimentSpec(exp)
	requireValid(t, result)
}

// ============================================================================
// E1–E4: Model Identity Edge Cases
// ============================================================================

func TestModelName_E1_EmptyString(t *testing.T) {
	exp := validExperiment()
	exp.Model.ModelName = ""
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "model_name")
}

func TestModelName_E2_PathTraversal(t *testing.T) {
	exp := validExperiment()
	exp.Model.ModelName = "../../etc/passwd"
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "model_name")
}

func TestModelName_E2_SpecialChars(t *testing.T) {
	exp := validExperiment()
	exp.Model.ModelName = "model with spaces"
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "model_name")
}

func TestModelName_E2_StartWithHyphen(t *testing.T) {
	exp := validExperiment()
	exp.Model.ModelName = "-invalid-start"
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "model_name")
}

func TestModelName_E4_ExceedsMaxLength(t *testing.T) {
	exp := validExperiment()
	exp.Model.ModelName = strings.Repeat("a", MaxModelNameLen+1)
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "model_name")
}

func TestModelName_ValidWithDotsAndHyphens(t *testing.T) {
	exp := validExperiment()
	exp.Model.ModelName = "llama-3.1-8b_instruct"
	result := ValidateExperimentSpec(exp)
	requireValid(t, result)
}

func TestModelName_ValidExactMaxLength(t *testing.T) {
	exp := validExperiment()
	// 1 leading alphanum + (MaxModelNameLen - 1) trailing chars = exactly max length.
	exp.Model.ModelName = "a" + strings.Repeat("b", MaxModelNameLen-1)
	result := ValidateExperimentSpec(exp)
	requireValid(t, result)
}

func TestModelSpec_ZeroLayers(t *testing.T) {
	exp := validExperiment()
	exp.Model.NumLayers = 0
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "num_layers")
}

func TestModelSpec_ZeroKVHeads(t *testing.T) {
	exp := validExperiment()
	exp.Model.NumKVHeads = 0
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "num_kv_heads")
}

func TestModelSpec_ZeroHeadDim(t *testing.T) {
	exp := validExperiment()
	exp.Model.HeadDim = 0
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "head_dim")
}

func TestModelSpec_ZeroMaxSeqLen(t *testing.T) {
	exp := validExperiment()
	exp.Model.MaxSeqLen = 0
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "max_seq_len")
}

// ============================================================================
// E5–E10: Tensor Parallelism Edge Cases
// ============================================================================

func TestTP_E5_EmptyList(t *testing.T) {
	exp := validExperiment()
	exp.Sweep.TPDegrees = []uint32{}
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "tp_degrees")
}

func TestTP_E6_ZeroValue(t *testing.T) {
	exp := validExperiment()
	exp.Sweep.TPDegrees = []uint32{0}
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "tp_degrees")
}

func TestTP_E7_NonPowerOfTwo(t *testing.T) {
	nonPow2 := []uint32{3, 5, 6, 7, 9, 10, 12, 15}
	for _, tp := range nonPow2 {
		t.Run(fmt.Sprintf("TP=%d", tp), func(t *testing.T) {
			exp := validExperiment()
			exp.Sweep.TPDegrees = []uint32{tp}
			result := ValidateExperimentSpec(exp)
			requireErrors(t, result, "tp_degrees")
		})
	}
}

func TestTP_E8_LargeTP16_Accepted(t *testing.T) {
	// TP=16 is a valid power of 2; hardware check is deferred to scheduler.
	exp := validExperiment()
	exp.Sweep.TPDegrees = []uint32{16}
	result := ValidateExperimentSpec(exp)
	requireValid(t, result)
}

func TestTP_ValidPowersOfTwo(t *testing.T) {
	valid := []uint32{1, 2, 4, 8}
	for _, tp := range valid {
		t.Run(fmt.Sprintf("TP=%d", tp), func(t *testing.T) {
			exp := validExperiment()
			exp.Sweep.TPDegrees = []uint32{tp}
			result := ValidateExperimentSpec(exp)
			requireValid(t, result)
		})
	}
}

// ============================================================================
// E11–E14: Batch Size Edge Cases
// ============================================================================

func TestBatchSize_E11_EmptyList(t *testing.T) {
	exp := validExperiment()
	exp.Sweep.BatchSizes = []uint32{}
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "batch_sizes")
}

func TestBatchSize_E12_ZeroValue(t *testing.T) {
	exp := validExperiment()
	exp.Sweep.BatchSizes = []uint32{0}
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "batch_sizes")
}

func TestBatchSize_E13_VeryLargeValue_Accepted(t *testing.T) {
	exp := validExperiment()
	exp.Sweep.BatchSizes = []uint32{4096}
	result := ValidateExperimentSpec(exp)
	requireValid(t, result)
}

// ============================================================================
// E15–E16: Quantization Edge Cases
// ============================================================================

func TestQuantization_E15_EmptyDefaultsToFP16(t *testing.T) {
	exp := validExperiment()
	exp.Sweep.Quantizations = []string{} // Should default, not error.
	result := ValidateExperimentSpec(exp)
	requireValid(t, result)
}

func TestQuantization_E16_Unspecified_Rejected(t *testing.T) {
	exp := validExperiment()
	exp.Sweep.Quantizations = []string{"QUANTIZATION_UNSPECIFIED"}
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "quantizations")
}

func TestQuantization_E16_EmptyString_Rejected(t *testing.T) {
	exp := validExperiment()
	exp.Sweep.Quantizations = []string{""}
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "quantizations")
}

// ============================================================================
// E17–E18: Backend Selection Edge Cases
// ============================================================================

func TestBackend_E17_Unspecified_Accepted(t *testing.T) {
	exp := validExperiment()
	exp.Backend = "BACKEND_UNSPECIFIED" // Will default to TRT_LLM.
	result := ValidateExperimentSpec(exp)
	requireValid(t, result)
}

func TestBackend_Unknown_Rejected(t *testing.T) {
	exp := validExperiment()
	exp.Backend = "ONNX_RUNTIME" // Not a valid backend.
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "backend")
}

func TestBackend_VLLM_Accepted(t *testing.T) {
	exp := validExperiment()
	exp.Backend = "VLLM"
	result := ValidateExperimentSpec(exp)
	requireValid(t, result)
}

// ============================================================================
// E19–E20: Sweep Combinatorial Explosion
// ============================================================================

func TestSweep_E19_TooManyCells(t *testing.T) {
	exp := validExperiment()
	// 4 TP × 10 batch × 7 quant = 280 > 256.
	exp.Sweep.TPDegrees = []uint32{1, 2, 4, 8}
	exp.Sweep.BatchSizes = []uint32{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
	exp.Sweep.Quantizations = []string{"FP16", "BF16", "FP8_E4M3", "FP8_E5M2", "INT4_AWQ", "INT4_GPTQ", "FP16"}
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "spec.sweep")
}

func TestSweep_E20_SinglePointBenchmark(t *testing.T) {
	exp := validExperiment()
	exp.Sweep.TPDegrees = []uint32{1}
	exp.Sweep.BatchSizes = []uint32{32}
	exp.Sweep.Quantizations = []string{"FP16"}
	result := ValidateExperimentSpec(exp)
	requireValid(t, result)
}

// ============================================================================
// E21–E29: Traffic Profile Edge Cases
// ============================================================================

func TestTraffic_E21_ZeroConcurrency_WithNoQPS(t *testing.T) {
	exp := validExperiment()
	exp.Traffic.Concurrency = 0
	exp.Traffic.TargetQPS = 0
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "spec.traffic")
}

func TestTraffic_E22_ExcessiveConcurrency(t *testing.T) {
	exp := validExperiment()
	exp.Traffic.Concurrency = 10_000
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "concurrency")
}

func TestTraffic_E23_ZeroDuration(t *testing.T) {
	exp := validExperiment()
	exp.Traffic.DurationSeconds = 0
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "duration")
}

func TestTraffic_E23_ShortDuration(t *testing.T) {
	exp := validExperiment()
	exp.Traffic.DurationSeconds = 5 // 5s < 10s minimum.
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "duration")
}

func TestTraffic_E24_VeryLongDuration_Warning(t *testing.T) {
	exp := validExperiment()
	exp.Traffic.DurationSeconds = 7200 // 2 hours > 1 hour warning threshold.
	result := ValidateExperimentSpec(exp)
	requireValid(t, result)         // Should NOT error.
	requireWarnings(t, result, "duration") // Should warn.
}

func TestTraffic_E25_ZeroPromptLength(t *testing.T) {
	exp := validExperiment()
	exp.Traffic.PromptLength = 0
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "prompt_length")
}

func TestTraffic_E27_ZeroOutputTokens(t *testing.T) {
	exp := validExperiment()
	exp.Traffic.MaxOutputTokens = 0
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "max_output_tokens")
}

func TestTraffic_E29_MutuallyExclusive_ConcurrencyAndQPS(t *testing.T) {
	exp := validExperiment()
	exp.Traffic.Concurrency = 32
	exp.Traffic.TargetQPS = 100.0
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "spec.traffic")
}

func TestTraffic_OpenLoopQPS_Valid(t *testing.T) {
	exp := validExperiment()
	exp.Traffic.Concurrency = 0
	exp.Traffic.TargetQPS = 100.0
	result := ValidateExperimentSpec(exp)
	requireValid(t, result)
}

// ============================================================================
// E30–E35: Performance Budget Edge Cases
// ============================================================================

func TestBudget_E30_ZeroTTFT(t *testing.T) {
	exp := validExperiment()
	b := validBudget()
	b.MaxP99TTFT_MS = 0
	exp.Budget = &b
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "max_p99_ttft_ms")
}

func TestBudget_E30_NegativeTTFT(t *testing.T) {
	exp := validExperiment()
	b := validBudget()
	b.MaxP99TTFT_MS = -10.0
	exp.Budget = &b
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "max_p99_ttft_ms")
}

func TestBudget_E31_ZeroITL(t *testing.T) {
	exp := validExperiment()
	b := validBudget()
	b.MaxP95ITL_MS = 0
	exp.Budget = &b
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "max_p95_itl_ms")
}

func TestBudget_E32_ZeroThroughput(t *testing.T) {
	exp := validExperiment()
	b := validBudget()
	b.MinThroughputTPS = 0
	exp.Budget = &b
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "min_throughput")
}

func TestBudget_E33_ThresholdOver100(t *testing.T) {
	exp := validExperiment()
	b := validBudget()
	b.RegressionThresholdPct = 150.0
	exp.Budget = &b
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "regression_threshold")
}

func TestBudget_E34_ThresholdZero(t *testing.T) {
	exp := validExperiment()
	b := validBudget()
	b.RegressionThresholdPct = 0
	exp.Budget = &b
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "regression_threshold")
}

func TestBudget_E35_CIModeWithoutBudget(t *testing.T) {
	exp := validExperiment()
	exp.CIMode = true
	exp.Budget = nil
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "ci_mode")
}

func TestBudget_CIModeWithValidBudget(t *testing.T) {
	exp := validExperiment()
	exp.CIMode = true
	b := validBudget()
	exp.Budget = &b
	result := ValidateExperimentSpec(exp)
	requireValid(t, result)
}

// ============================================================================
// KV Cache Memory Fraction Edge Cases
// ============================================================================

func TestKVCacheFraction_TooLow(t *testing.T) {
	exp := validExperiment()
	exp.Sweep.KVCacheMemFraction = 0.05 // Below 0.1 minimum.
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "kv_cache_mem_fraction")
}

func TestKVCacheFraction_TooHigh(t *testing.T) {
	exp := validExperiment()
	exp.Sweep.KVCacheMemFraction = 1.0 // Above 0.99 maximum.
	result := ValidateExperimentSpec(exp)
	requireErrors(t, result, "kv_cache_mem_fraction")
}

func TestKVCacheFraction_Zero_UsesDefault(t *testing.T) {
	exp := validExperiment()
	exp.Sweep.KVCacheMemFraction = 0.0 // Zero means "use default 0.85".
	result := ValidateExperimentSpec(exp)
	requireValid(t, result)
}

// ============================================================================
// Helper Function Tests
// ============================================================================

func TestIsPowerOfTwo(t *testing.T) {
	tests := []struct {
		input    uint32
		expected bool
	}{
		{0, false},
		{1, true},
		{2, true},
		{3, false},
		{4, true},
		{5, false},
		{6, false},
		{7, false},
		{8, true},
		{16, true},
		{32, true},
		{64, true},
		{100, false},
		{128, true},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%d", tc.input), func(t *testing.T) {
			got := isPowerOfTwo(tc.input)
			if got != tc.expected {
				t.Errorf("isPowerOfTwo(%d) = %v, want %v", tc.input, got, tc.expected)
			}
		})
	}
}

func TestDeduplicateUint32(t *testing.T) {
	tests := []struct {
		name     string
		input    []uint32
		expected []uint32
	}{
		{"empty", nil, []uint32{}},
		{"no_dups", []uint32{1, 2, 4}, []uint32{1, 2, 4}},
		{"with_dups_E9", []uint32{2, 4, 2}, []uint32{2, 4}},
		{"all_same", []uint32{8, 8, 8}, []uint32{8}},
		{"preserves_order", []uint32{4, 1, 2, 1, 4}, []uint32{4, 1, 2}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := DeduplicateUint32(tc.input)
			if len(got) != len(tc.expected) {
				t.Fatalf("len = %d, want %d: got %v", len(got), len(tc.expected), got)
			}
			for i := range got {
				if got[i] != tc.expected[i] {
					t.Errorf("index %d: got %d, want %d", i, got[i], tc.expected[i])
				}
			}
		})
	}
}

func TestEstimateKVCacheBytesPerToken(t *testing.T) {
	// Llama-3-8B: 2 × 32 layers × 8 KV heads × 128 head_dim × 2 bytes = 131,072 bytes.
	got := EstimateKVCacheBytesPerToken(32, 8, 128, 2)
	expected := uint64(131_072)
	if got != expected {
		t.Errorf("KV cache bytes/token = %d, want %d", got, expected)
	}

	// Same model with FP8: 2 × 32 × 8 × 128 × 1 = 65,536 bytes.
	got = EstimateKVCacheBytesPerToken(32, 8, 128, 1)
	expected = uint64(65_536)
	if got != expected {
		t.Errorf("KV cache bytes/token (FP8) = %d, want %d", got, expected)
	}
}

func TestEstimateWeightMemoryBytes(t *testing.T) {
	// Llama-3-8B FP16: 8.03B × 2 bytes = 16,060,000,000 bytes.
	got := EstimateWeightMemoryBytes(8_030_000_000, 2.0)
	expected := uint64(16_060_000_000)
	if got != expected {
		t.Errorf("weight memory = %d, want %d", got, expected)
	}

	// INT4: 8.03B × 0.5 bytes = 4,015,000,000 bytes.
	got = EstimateWeightMemoryBytes(8_030_000_000, 0.5)
	expected = uint64(4_015_000_000)
	if got != expected {
		t.Errorf("weight memory (INT4) = %d, want %d", got, expected)
	}
}

// ============================================================================
// Compound / Integration Edge Cases
// ============================================================================

func TestMultipleErrors_ReportedTogether(t *testing.T) {
	// Submit an experiment with MANY problems — all should be reported in one pass.
	exp := ExperimentInput{
		Model: ModelSpecInput{
			ModelName: "", // E1
			// All zeros — E1 + missing fields.
		},
		Sweep: SweepInput{
			TPDegrees:  []uint32{}, // E5
			BatchSizes: []uint32{}, // E11
		},
		Traffic: TrafficInput{
			Concurrency:     0,
			TargetQPS:       0, // Both zero.
			DurationSeconds: 0, // E23
			PromptLength:    0, // E25
			MaxOutputTokens: 0, // E27
		},
		CIMode: true,
		Budget: nil, // E35
	}

	result := ValidateExperimentSpec(exp)
	if result.IsValid() {
		t.Fatal("expected multiple errors, got none")
	}

	// Should catch ALL problems, not just the first one.
	if len(result.Errors) < 5 {
		t.Errorf("expected at least 5 errors, got %d:\n%s", len(result.Errors), result.ErrorMessages())
	}
}
