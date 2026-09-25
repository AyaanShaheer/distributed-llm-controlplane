package orchestrator

import (
	"fmt"
	"testing"
)

// ============================================================================
// SW1: Minimal Single-Point Sweep
// ============================================================================

func TestSW1_MinimalSinglePoint(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:  []uint32{1},
		BatchSizes: []uint32{32},
		Quantizations: []string{"FP16"},
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.TotalCells() != 1 {
		t.Errorf("expected 1 cell, got %d", plan.TotalCells())
	}
	cell := plan.Cells[0]
	if cell.TPDegree != 1 || cell.BatchSize != 32 || cell.Quantization != "FP16" {
		t.Errorf("unexpected cell: %s", cell)
	}
	if cell.Index != 0 {
		t.Errorf("first cell index = %d, want 0", cell.Index)
	}
}

// ============================================================================
// SW2: Standard Grid (2×3×1 = 6 cells)
// ============================================================================

func TestSW2_StandardGrid(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{1, 2},
		BatchSizes:    []uint32{1, 8, 32},
		Quantizations: []string{"FP16"},
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.TotalCells() != 6 {
		t.Fatalf("expected 6 cells, got %d", plan.TotalCells())
	}

	// Verify TP ordering: first 3 cells should be TP=1, next 3 should be TP=2.
	for i := 0; i < 3; i++ {
		if plan.Cells[i].TPDegree != 1 {
			t.Errorf("cell[%d].TP = %d, want 1", i, plan.Cells[i].TPDegree)
		}
	}
	for i := 3; i < 6; i++ {
		if plan.Cells[i].TPDegree != 2 {
			t.Errorf("cell[%d].TP = %d, want 2", i, plan.Cells[i].TPDegree)
		}
	}

	// Verify batch ordering within TP groups: 1, 8, 32.
	expectedBatch := []uint32{1, 8, 32, 1, 8, 32}
	for i, expected := range expectedBatch {
		if plan.Cells[i].BatchSize != expected {
			t.Errorf("cell[%d].Batch = %d, want %d", i, plan.Cells[i].BatchSize, expected)
		}
	}
}

// ============================================================================
// SW3: Full Grid (4×4×2 = 32 cells)
// ============================================================================

func TestSW3_FullGrid(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{1, 2, 4, 8},
		BatchSizes:    []uint32{1, 8, 32, 128},
		Quantizations: []string{"FP16", "FP8_E4M3"},
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.TotalCells() != 32 {
		t.Fatalf("expected 32 cells, got %d", plan.TotalCells())
	}

	// P1: All cells must be unique.
	seen := make(map[string]bool)
	for _, cell := range plan.Cells {
		key := cell.Key()
		if seen[key] {
			t.Errorf("duplicate cell: %s", key)
		}
		seen[key] = true
	}
}

// ============================================================================
// SW4: Empty Quantizations Defaults to FP16
// ============================================================================

func TestSW4_EmptyQuantDefaultsToFP16(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{1},
		BatchSizes:    []uint32{1},
		Quantizations: []string{}, // Empty → default to FP16.
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.TotalCells() != 1 {
		t.Fatalf("expected 1 cell, got %d", plan.TotalCells())
	}
	if plan.Cells[0].Quantization != "FP16" {
		t.Errorf("quantization = %q, want FP16", plan.Cells[0].Quantization)
	}
}

// ============================================================================
// SW5: Duplicate TP Degrees Deduplicated
// ============================================================================

func TestSW5_DuplicateTP(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{2, 4, 2, 4},
		BatchSizes:    []uint32{1},
		Quantizations: []string{"FP16"},
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.TotalCells() != 2 {
		t.Errorf("expected 2 cells after dedup, got %d", plan.TotalCells())
	}
	if len(plan.Dimensions.TPDegrees) != 2 {
		t.Errorf("expected 2 unique TP degrees, got %d", len(plan.Dimensions.TPDegrees))
	}
}

// ============================================================================
// SW6: Duplicate Batch Sizes Deduplicated
// ============================================================================

func TestSW6_DuplicateBatch(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{1},
		BatchSizes:    []uint32{8, 8, 8},
		Quantizations: []string{"FP16"},
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.TotalCells() != 1 {
		t.Errorf("expected 1 cell after dedup, got %d", plan.TotalCells())
	}
}

// ============================================================================
// SW7: Duplicate Quantizations Deduplicated
// ============================================================================

func TestSW7_DuplicateQuant(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{1},
		BatchSizes:    []uint32{1},
		Quantizations: []string{"FP16", "FP16", "BF16"},
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.TotalCells() != 2 {
		t.Errorf("expected 2 cells after dedup, got %d", plan.TotalCells())
	}
}

// ============================================================================
// SW8: TP Degrees Sorted Ascending
// ============================================================================

func TestSW8_TPSortedAscending(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{8, 2, 4, 1},
		BatchSizes:    []uint32{1},
		Quantizations: []string{"FP16"},
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectedTP := []uint32{1, 2, 4, 8}
	for i, expected := range expectedTP {
		if plan.Cells[i].TPDegree != expected {
			t.Errorf("cell[%d].TP = %d, want %d", i, plan.Cells[i].TPDegree, expected)
		}
	}
	// Also verify resolved dimensions.
	for i, expected := range expectedTP {
		if plan.Dimensions.TPDegrees[i] != expected {
			t.Errorf("Dimensions.TP[%d] = %d, want %d", i, plan.Dimensions.TPDegrees[i], expected)
		}
	}
}

// ============================================================================
// SW9: Batch Sizes Sorted Ascending
// ============================================================================

func TestSW9_BatchSortedAscending(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{1},
		BatchSizes:    []uint32{128, 1, 32, 8},
		Quantizations: []string{"FP16"},
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectedBatch := []uint32{1, 8, 32, 128}
	for i, expected := range expectedBatch {
		if plan.Cells[i].BatchSize != expected {
			t.Errorf("cell[%d].Batch = %d, want %d", i, plan.Cells[i].BatchSize, expected)
		}
	}
}

// ============================================================================
// SW10: Maximum Cells Exactly at Limit
// ============================================================================

func TestSW10_ExactlyMaxCells(t *testing.T) {
	// 4 TP × 8 Batch × 8 Quant = 256 = MaxSweepCells.
	tps := []uint32{1, 2, 4, 8}
	batches := []uint32{1, 2, 4, 8, 16, 32, 64, 128}
	quants := []string{"FP16", "BF16", "FP8_E4M3", "FP8_E5M2", "INT4_AWQ", "INT4_GPTQ", "Q1", "Q2"}

	sweep := SweepInput{
		TPDegrees:     tps,
		BatchSizes:    batches,
		Quantizations: quants,
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("256 cells should be accepted: %v", err)
	}
	if plan.TotalCells() != 256 {
		t.Errorf("expected 256 cells, got %d", plan.TotalCells())
	}
}

// ============================================================================
// SW11: Exceeds Maximum Cells
// ============================================================================

func TestSW11_ExceedsMaxCells(t *testing.T) {
	// 4 TP × 10 Batch × 7 Quant = 280 > 256.
	tps := []uint32{1, 2, 4, 8}
	batches := []uint32{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
	quants := []string{"FP16", "BF16", "FP8_E4M3", "FP8_E5M2", "INT4_AWQ", "INT4_GPTQ", "Q1"}

	sweep := SweepInput{
		TPDegrees:     tps,
		BatchSizes:    batches,
		Quantizations: quants,
	}
	_, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err == nil {
		t.Fatal("expected error for 280 cells exceeding limit of 256")
	}
}

// ============================================================================
// SW12: KV Cache Fraction Defaults to 0.85
// ============================================================================

func TestSW12_KVCacheFractionDefault(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:          []uint32{1},
		BatchSizes:         []uint32{1},
		Quantizations:      []string{"FP16"},
		KVCacheMemFraction: 0.0, // Zero → default.
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.ResolvedKVCacheMemFraction != DefaultKVCacheMemFraction {
		t.Errorf("KV fraction = %f, want %f", plan.ResolvedKVCacheMemFraction, DefaultKVCacheMemFraction)
	}
}

// ============================================================================
// SW13: KV Cache Fraction Preserves User Value
// ============================================================================

func TestSW13_KVCacheFractionUserValue(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:          []uint32{1},
		BatchSizes:         []uint32{1},
		Quantizations:      []string{"FP16"},
		KVCacheMemFraction: 0.92,
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.ResolvedKVCacheMemFraction != 0.92 {
		t.Errorf("KV fraction = %f, want 0.92", plan.ResolvedKVCacheMemFraction)
	}
}

// ============================================================================
// SW14: Empty TP Degrees Rejected
// ============================================================================

func TestSW14_EmptyTPRejected(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{},
		BatchSizes:    []uint32{1},
		Quantizations: []string{"FP16"},
	}
	_, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err == nil {
		t.Fatal("expected error for empty TP degrees")
	}
}

// ============================================================================
// SW15: Empty Batch Sizes Rejected
// ============================================================================

func TestSW15_EmptyBatchRejected(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{1},
		BatchSizes:    []uint32{},
		Quantizations: []string{"FP16"},
	}
	_, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err == nil {
		t.Fatal("expected error for empty batch sizes")
	}
}

// ============================================================================
// SW16: Backend Defaults from UNSPECIFIED to TRT_LLM
// ============================================================================

func TestSW16_BackendDefaultToTRTLLM(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{1},
		BatchSizes:    []uint32{1},
		Quantizations: []string{"FP16"},
	}
	plan, err := GenerateExecutionPlan(sweep, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.ResolvedBackend != "TRT_LLM" {
		t.Errorf("backend = %q, want TRT_LLM", plan.ResolvedBackend)
	}
}

func TestSW16_BackendDefaultFromUnspecified(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{1},
		BatchSizes:    []uint32{1},
		Quantizations: []string{"FP16"},
	}
	plan, err := GenerateExecutionPlan(sweep, "BACKEND_UNSPECIFIED")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.ResolvedBackend != "TRT_LLM" {
		t.Errorf("backend = %q, want TRT_LLM", plan.ResolvedBackend)
	}
}

// ============================================================================
// P1: Cell Uniqueness Property
// ============================================================================

func TestP1_CellUniqueness(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{1, 2, 4},
		BatchSizes:    []uint32{1, 8, 32, 128},
		Quantizations: []string{"FP16", "FP8_E4M3"},
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	seen := make(map[string]int)
	for _, cell := range plan.Cells {
		key := cell.Key()
		if prev, ok := seen[key]; ok {
			t.Errorf("duplicate cell at indices %d and %d: %s", prev, cell.Index, key)
		}
		seen[key] = cell.Index
	}
}

// ============================================================================
// P2: TP Ordering Property
// ============================================================================

func TestP2_TPOrdering(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{4, 1, 8, 2},
		BatchSizes:    []uint32{1, 32},
		Quantizations: []string{"FP16"},
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// TP values should never decrease across sequential cells.
	for i := 1; i < len(plan.Cells); i++ {
		if plan.Cells[i].TPDegree < plan.Cells[i-1].TPDegree {
			t.Errorf("TP ordering violated at cell[%d]: TP=%d followed by TP=%d",
				i, plan.Cells[i-1].TPDegree, plan.Cells[i].TPDegree)
		}
	}
}

// ============================================================================
// P4: Cell Count Matches Deduplicated Product
// ============================================================================

func TestP4_CellCount(t *testing.T) {
	// Input with duplicates: 3 unique TP × 2 unique Batch × 2 unique Quant = 12.
	sweep := SweepInput{
		TPDegrees:     []uint32{1, 2, 4, 2, 1},     // 3 unique.
		BatchSizes:    []uint32{8, 32, 8},           // 2 unique.
		Quantizations: []string{"FP16", "BF16", "FP16"}, // 2 unique.
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := 3 * 2 * 2
	if plan.TotalCells() != expected {
		t.Errorf("cell count = %d, want %d (3×2×2)", plan.TotalCells(), expected)
	}
}

// ============================================================================
// P5: Sequential Cell Indices
// ============================================================================

func TestP5_SequentialIndices(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{1, 2},
		BatchSizes:    []uint32{1, 8, 32},
		Quantizations: []string{"FP16"},
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, cell := range plan.Cells {
		if cell.Index != i {
			t.Errorf("cell[%d].Index = %d, want %d", i, cell.Index, i)
		}
	}
}

// ============================================================================
// P3: Within-TP Ordering (Quant then Batch)
// ============================================================================

func TestP3_WithinTPOrdering(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{1},
		BatchSizes:    []uint32{32, 1, 8},
		Quantizations: []string{"FP16", "BF16"},
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Expected order (all TP=1):
	// BF16-1, BF16-8, BF16-32, FP16-1, FP16-8, FP16-32
	expected := []struct {
		quant string
		batch uint32
	}{
		{"BF16", 1}, {"BF16", 8}, {"BF16", 32},
		{"FP16", 1}, {"FP16", 8}, {"FP16", 32},
	}

	if len(plan.Cells) != len(expected) {
		t.Fatalf("expected %d cells, got %d", len(expected), len(plan.Cells))
	}
	for i, exp := range expected {
		cell := plan.Cells[i]
		if cell.Quantization != exp.quant || cell.BatchSize != exp.batch {
			t.Errorf("cell[%d] = {%s, %d}, want {%s, %d}",
				i, cell.Quantization, cell.BatchSize, exp.quant, exp.batch)
		}
	}
}

// ============================================================================
// SweepCell String & Key
// ============================================================================

func TestSweepCell_String(t *testing.T) {
	cell := SweepCell{Index: 5, TPDegree: 4, BatchSize: 128, Quantization: "FP8_E4M3"}
	s := cell.String()
	if s != "cell[5]{TP=4, Batch=128, Quant=FP8_E4M3}" {
		t.Errorf("String() = %q", s)
	}
}

func TestSweepCell_Key(t *testing.T) {
	cell := SweepCell{TPDegree: 2, BatchSize: 64, Quantization: "FP16"}
	if cell.Key() != "2:64:FP16" {
		t.Errorf("Key() = %q", cell.Key())
	}
}

// ============================================================================
// deduplicateStrings Helper
// ============================================================================

func TestDeduplicateStrings(t *testing.T) {
	tests := []struct {
		name     string
		input    []string
		expected []string
	}{
		{"empty", nil, []string{}},
		{"no_dups", []string{"FP16", "BF16"}, []string{"FP16", "BF16"}},
		{"with_dups", []string{"FP16", "FP16", "BF16"}, []string{"FP16", "BF16"}},
		{"all_same", []string{"FP16", "FP16", "FP16"}, []string{"FP16"}},
		{"preserves_first_occurrence_order", []string{"BF16", "FP16", "BF16"}, []string{"BF16", "FP16"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := deduplicateStrings(tc.input)
			if len(got) != len(tc.expected) {
				t.Fatalf("len = %d, want %d: got %v", len(got), len(tc.expected), got)
			}
			for i := range got {
				if got[i] != tc.expected[i] {
					t.Errorf("index %d: got %q, want %q", i, got[i], tc.expected[i])
				}
			}
		})
	}
}

// ============================================================================
// VLLM Backend Preserved
// ============================================================================

func TestBackendVLLM_Preserved(t *testing.T) {
	sweep := SweepInput{
		TPDegrees:     []uint32{1},
		BatchSizes:    []uint32{1},
		Quantizations: []string{"FP16"},
	}
	plan, err := GenerateExecutionPlan(sweep, "VLLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plan.ResolvedBackend != "VLLM" {
		t.Errorf("backend = %q, want VLLM", plan.ResolvedBackend)
	}
}

// ============================================================================
// Large Sweep With Dedup Stays Under Limit
// ============================================================================

func TestLargeSweepWithDedupUnderLimit(t *testing.T) {
	// Raw: 8 TP × 10 Batch × 7 Quant = 560 > 256.
	// But with heavy duplication, unique = 4 TP × 5 Batch × 3 Quant = 60 ≤ 256.
	tps := []uint32{1, 2, 4, 8, 1, 2, 4, 8}
	batches := []uint32{1, 8, 32, 64, 128, 1, 8, 32, 64, 128}
	quants := []string{"FP16", "BF16", "FP8_E4M3", "FP16", "BF16", "FP8_E4M3", "FP16"}

	sweep := SweepInput{
		TPDegrees:     tps,
		BatchSizes:    batches,
		Quantizations: quants,
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("should succeed after dedup: %v", err)
	}
	if plan.TotalCells() != 60 {
		t.Errorf("expected 60 cells (4×5×3), got %d", plan.TotalCells())
	}
}

// ============================================================================
// Stress: Generate and Verify 256-Cell Plan
// ============================================================================

func TestStress_VerifyFullPlan(t *testing.T) {
	tps := []uint32{1, 2, 4, 8}
	batches := []uint32{1, 2, 4, 8, 16, 32, 64, 128}
	quants := []string{"FP16", "BF16", "FP8_E4M3", "FP8_E5M2", "INT4_AWQ", "INT4_GPTQ", "Q1", "Q2"}

	sweep := SweepInput{
		TPDegrees:     tps,
		BatchSizes:    batches,
		Quantizations: quants,
	}
	plan, err := GenerateExecutionPlan(sweep, "TRT_LLM")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify all 256 cells exist with unique keys.
	keys := make(map[string]bool)
	for _, cell := range plan.Cells {
		key := cell.Key()
		if keys[key] {
			t.Errorf("duplicate key: %s", key)
		}
		keys[key] = true
	}
	if len(keys) != 256 {
		t.Errorf("expected 256 unique keys, got %d", len(keys))
	}

	// Verify every combination is present.
	for _, tp := range tps {
		for _, bs := range batches {
			for _, q := range quants {
				key := fmt.Sprintf("%d:%d:%s", tp, bs, q)
				if !keys[key] {
					t.Errorf("missing cell: %s", key)
				}
			}
		}
	}
}
