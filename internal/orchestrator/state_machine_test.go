package orchestrator

import (
	"fmt"
	"sync"
	"testing"
)

// ============================================================================
// State String Representation
// ============================================================================

func TestExperimentState_String(t *testing.T) {
	tests := []struct {
		state    ExperimentState
		expected string
	}{
		{StatePending, "PENDING"},
		{StateProvisioning, "PROVISIONING"},
		{StateDeploying, "DEPLOYING"},
		{StateWarmingUp, "WARMING_UP"},
		{StateBenchmarking, "BENCHMARKING"},
		{StateCollecting, "COLLECTING"},
		{StateTearingDown, "TEARING_DOWN"},
		{StateCompleted, "COMPLETED"},
		{StateFailed, "FAILED"},
		{StateCancelled, "CANCELLED"},
		{ExperimentState(99), "UNKNOWN(99)"},
	}
	for _, tc := range tests {
		t.Run(tc.expected, func(t *testing.T) {
			if got := tc.state.String(); got != tc.expected {
				t.Errorf("String() = %q, want %q", got, tc.expected)
			}
		})
	}
}

// ============================================================================
// SM1–SM7: Valid Forward Transitions
// ============================================================================

func TestSM1_Pending_To_Provisioning(t *testing.T) {
	exp := NewExperiment("test-sm1", ExperimentInput{})
	if err := exp.TransitionTo(StateProvisioning, "allocating GPUs"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if exp.GetState() != StateProvisioning {
		t.Errorf("state = %s, want PROVISIONING", exp.GetState())
	}
}

func TestSM2_Provisioning_To_Deploying(t *testing.T) {
	exp := NewExperiment("test-sm2", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning)
	if err := exp.TransitionTo(StateDeploying, "launching Triton containers"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if exp.GetState() != StateDeploying {
		t.Errorf("state = %s, want DEPLOYING", exp.GetState())
	}
}

func TestSM3_Deploying_To_WarmingUp(t *testing.T) {
	exp := NewExperiment("test-sm3", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning, StateDeploying)
	if err := exp.TransitionTo(StateWarmingUp, "running warmup requests"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if exp.GetState() != StateWarmingUp {
		t.Errorf("state = %s, want WARMING_UP", exp.GetState())
	}
}

func TestSM4_WarmingUp_To_Benchmarking(t *testing.T) {
	exp := NewExperiment("test-sm4", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning, StateDeploying, StateWarmingUp)
	if err := exp.TransitionTo(StateBenchmarking, "starting load injection"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if exp.GetState() != StateBenchmarking {
		t.Errorf("state = %s, want BENCHMARKING", exp.GetState())
	}
}

func TestSM5_Benchmarking_To_Collecting(t *testing.T) {
	exp := NewExperiment("test-sm5", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning, StateDeploying, StateWarmingUp, StateBenchmarking)
	if err := exp.TransitionTo(StateCollecting, "draining in-flight requests"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if exp.GetState() != StateCollecting {
		t.Errorf("state = %s, want COLLECTING", exp.GetState())
	}
}

func TestSM6_Collecting_To_TearingDown(t *testing.T) {
	exp := NewExperiment("test-sm6", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning, StateDeploying, StateWarmingUp,
		StateBenchmarking, StateCollecting)
	if err := exp.TransitionTo(StateTearingDown, "releasing GPU allocation"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if exp.GetState() != StateTearingDown {
		t.Errorf("state = %s, want TEARING_DOWN", exp.GetState())
	}
}

func TestSM7_TearingDown_To_Completed(t *testing.T) {
	exp := NewExperiment("test-sm7", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning, StateDeploying, StateWarmingUp,
		StateBenchmarking, StateCollecting, StateTearingDown)
	if err := exp.TransitionTo(StateCompleted, "experiment finished"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if exp.GetState() != StateCompleted {
		t.Errorf("state = %s, want COMPLETED", exp.GetState())
	}
}

// ============================================================================
// SM8: Full Lifecycle Traversal
// ============================================================================

func TestSM8_FullLifecycle(t *testing.T) {
	exp := NewExperiment("test-sm8", ExperimentInput{})
	states := []ExperimentState{
		StateProvisioning, StateDeploying, StateWarmingUp,
		StateBenchmarking, StateCollecting, StateTearingDown, StateCompleted,
	}
	for _, s := range states {
		if err := exp.TransitionTo(s, "lifecycle step"); err != nil {
			t.Fatalf("transition to %s failed: %v", s, err)
		}
	}
	if exp.GetState() != StateCompleted {
		t.Errorf("final state = %s, want COMPLETED", exp.GetState())
	}
	// Verify audit log has exactly 7 transitions.
	log := exp.GetStateLog()
	if len(log) != 7 {
		t.Errorf("state log has %d entries, want 7", len(log))
	}
}

// ============================================================================
// SM9–SM13: Invalid Forward Transitions (Skip & Backward)
// ============================================================================

func TestSM9_Skip_Pending_To_Benchmarking(t *testing.T) {
	exp := NewExperiment("test-sm9", ExperimentInput{})
	err := exp.TransitionTo(StateBenchmarking, "skip attempt")
	assertTransitionError(t, err)
	if exp.GetState() != StatePending {
		t.Errorf("state should remain PENDING, got %s", exp.GetState())
	}
}

func TestSM10_Skip_Pending_To_Deploying(t *testing.T) {
	exp := NewExperiment("test-sm10", ExperimentInput{})
	err := exp.TransitionTo(StateDeploying, "skip attempt")
	assertTransitionError(t, err)
}

func TestSM11_Skip_Provisioning_To_WarmingUp(t *testing.T) {
	exp := NewExperiment("test-sm11", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning)
	err := exp.TransitionTo(StateWarmingUp, "skip attempt")
	assertTransitionError(t, err)
	if exp.GetState() != StateProvisioning {
		t.Errorf("state should remain PROVISIONING, got %s", exp.GetState())
	}
}

func TestSM12_Backward_Benchmarking_To_WarmingUp(t *testing.T) {
	exp := NewExperiment("test-sm12", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning, StateDeploying, StateWarmingUp, StateBenchmarking)
	err := exp.TransitionTo(StateWarmingUp, "backward attempt")
	assertTransitionError(t, err)
	if exp.GetState() != StateBenchmarking {
		t.Errorf("state should remain BENCHMARKING, got %s", exp.GetState())
	}
}

func TestSM13_Backward_Deploying_To_Provisioning(t *testing.T) {
	exp := NewExperiment("test-sm13", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning, StateDeploying)
	err := exp.TransitionTo(StateProvisioning, "backward attempt")
	assertTransitionError(t, err)
}

// ============================================================================
// SM14–SM15: Self-Transitions
// ============================================================================

func TestSM14_SelfTransition_Pending(t *testing.T) {
	exp := NewExperiment("test-sm14", ExperimentInput{})
	err := exp.TransitionTo(StatePending, "self transition")
	assertTransitionError(t, err)
}

func TestSM15_SelfTransition_Benchmarking(t *testing.T) {
	exp := NewExperiment("test-sm15", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning, StateDeploying, StateWarmingUp, StateBenchmarking)
	err := exp.TransitionTo(StateBenchmarking, "self transition")
	assertTransitionError(t, err)
}

// ============================================================================
// SM-T1–SM-T9: Terminal State Invariants
// ============================================================================

func TestSMT1_Completed_To_Benchmarking(t *testing.T) {
	exp := completedExperiment(t)
	err := exp.TransitionTo(StateBenchmarking, "revert attempt")
	assertTransitionError(t, err)
	if exp.GetState() != StateCompleted {
		t.Errorf("state should remain COMPLETED, got %s", exp.GetState())
	}
}

func TestSMT2_Completed_To_Failed(t *testing.T) {
	exp := completedExperiment(t)
	err := exp.TransitionTo(StateFailed, "fail completed")
	assertTransitionError(t, err)
}

func TestSMT3_Completed_To_Cancelled(t *testing.T) {
	exp := completedExperiment(t)
	err := exp.TransitionTo(StateCancelled, "cancel completed")
	assertTransitionError(t, err)
}

func TestSMT4_Completed_To_Completed(t *testing.T) {
	exp := completedExperiment(t)
	err := exp.TransitionTo(StateCompleted, "double complete")
	assertTransitionError(t, err)
}

func TestSMT5_Failed_To_Pending(t *testing.T) {
	exp := failedExperiment(t)
	err := exp.TransitionTo(StatePending, "restart attempt")
	assertTransitionError(t, err)
	if exp.GetState() != StateFailed {
		t.Errorf("state should remain FAILED, got %s", exp.GetState())
	}
}

func TestSMT6_Failed_To_Provisioning(t *testing.T) {
	exp := failedExperiment(t)
	err := exp.TransitionTo(StateProvisioning, "retry attempt")
	assertTransitionError(t, err)
}

func TestSMT7_Failed_To_Failed(t *testing.T) {
	exp := failedExperiment(t)
	err := exp.TransitionTo(StateFailed, "double fail")
	assertTransitionError(t, err)
}

func TestSMT8_Cancelled_To_Pending(t *testing.T) {
	exp := cancelledExperiment(t)
	err := exp.TransitionTo(StatePending, "restart after cancel")
	assertTransitionError(t, err)
}

func TestSMT9_Cancelled_To_Cancelled(t *testing.T) {
	exp := cancelledExperiment(t)
	err := exp.TransitionTo(StateCancelled, "double cancel")
	assertTransitionError(t, err)
}

// ============================================================================
// SM-F1–SM-F7: Failure From Every Non-Terminal State
// ============================================================================

func TestSMF_FailFromEveryNonTerminalState(t *testing.T) {
	nonTerminalStates := []struct {
		name   string
		setup  []ExperimentState
		state  ExperimentState
	}{
		{"SM-F1_PENDING", nil, StatePending},
		{"SM-F2_PROVISIONING", []ExperimentState{StateProvisioning}, StateProvisioning},
		{"SM-F3_DEPLOYING", []ExperimentState{StateProvisioning, StateDeploying}, StateDeploying},
		{"SM-F4_WARMING_UP", []ExperimentState{StateProvisioning, StateDeploying, StateWarmingUp}, StateWarmingUp},
		{"SM-F5_BENCHMARKING", []ExperimentState{StateProvisioning, StateDeploying, StateWarmingUp, StateBenchmarking}, StateBenchmarking},
		{"SM-F6_COLLECTING", []ExperimentState{StateProvisioning, StateDeploying, StateWarmingUp, StateBenchmarking, StateCollecting}, StateCollecting},
		{"SM-F7_TEARING_DOWN", []ExperimentState{StateProvisioning, StateDeploying, StateWarmingUp, StateBenchmarking, StateCollecting, StateTearingDown}, StateTearingDown},
	}

	for _, tc := range nonTerminalStates {
		t.Run(tc.name, func(t *testing.T) {
			exp := NewExperiment("test-"+tc.name, ExperimentInput{})
			for _, s := range tc.setup {
				mustTransition(t, exp, s)
			}
			if exp.GetState() != tc.state {
				t.Fatalf("setup state = %s, want %s", exp.GetState(), tc.state)
			}
			err := exp.TransitionTo(StateFailed, "CUDA OOM: batch_size=256 exceeded VRAM")
			if err != nil {
				t.Fatalf("transition to FAILED from %s should succeed: %v", tc.state, err)
			}
			if exp.GetState() != StateFailed {
				t.Errorf("state = %s, want FAILED", exp.GetState())
			}
			if exp.GetError() == "" {
				t.Error("error reason should be populated after FAILED transition")
			}
		})
	}
}

// ============================================================================
// SM-C1–SM-C5: Cancellation Edge Cases
// ============================================================================

func TestSMC1_Cancel_Pending(t *testing.T) {
	exp := NewExperiment("test-smc1", ExperimentInput{})
	cancelled, warning := exp.Cancel("user requested")
	if !cancelled {
		t.Error("expected cancellation to succeed")
	}
	if warning != "" {
		t.Errorf("unexpected warning: %s", warning)
	}
	if exp.GetState() != StateCancelled {
		t.Errorf("state = %s, want CANCELLED", exp.GetState())
	}
}

func TestSMC2_Cancel_Provisioning(t *testing.T) {
	exp := NewExperiment("test-smc2", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning)
	cancelled, _ := exp.Cancel("timeout waiting for GPU allocation")
	if !cancelled {
		t.Error("expected cancellation to succeed")
	}
	if exp.GetState() != StateCancelled {
		t.Errorf("state = %s, want CANCELLED", exp.GetState())
	}
}

func TestSMC3_Cancel_Benchmarking(t *testing.T) {
	exp := NewExperiment("test-smc3", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning, StateDeploying, StateWarmingUp, StateBenchmarking)
	cancelled, _ := exp.Cancel("user abort")
	if !cancelled {
		t.Error("expected cancellation to succeed")
	}
	if exp.GetState() != StateCancelled {
		t.Errorf("state = %s, want CANCELLED", exp.GetState())
	}
}

func TestSMC4_Cancel_TearingDown(t *testing.T) {
	exp := NewExperiment("test-smc4", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning, StateDeploying, StateWarmingUp,
		StateBenchmarking, StateCollecting, StateTearingDown)
	cancelled, _ := exp.Cancel("force cancel during teardown")
	if !cancelled {
		t.Error("expected cancellation to succeed")
	}
}

func TestSMC5_Cancel_AlreadyCompleted_E40(t *testing.T) {
	exp := completedExperiment(t)
	cancelled, warning := exp.Cancel("late cancel attempt")
	if cancelled {
		t.Error("should NOT cancel an already-completed experiment")
	}
	if warning == "" {
		t.Error("expected a warning message for cancel on completed experiment")
	}
	if exp.GetState() != StateCompleted {
		t.Errorf("state should remain COMPLETED, got %s", exp.GetState())
	}
}

func TestSMC5_Cancel_AlreadyFailed(t *testing.T) {
	exp := failedExperiment(t)
	cancelled, warning := exp.Cancel("cancel after failure")
	if cancelled {
		t.Error("should NOT cancel an already-failed experiment")
	}
	if warning == "" {
		t.Error("expected a warning")
	}
}

// ============================================================================
// ES8: Audit Log Records Every Transition
// ============================================================================

func TestES8_AuditLog(t *testing.T) {
	exp := NewExperiment("test-es8", ExperimentInput{})
	transitions := []struct {
		state  ExperimentState
		reason string
	}{
		{StateProvisioning, "allocating nodes"},
		{StateDeploying, "launching containers"},
		{StateWarmingUp, "warmup phase"},
		{StateBenchmarking, "load injection"},
		{StateCollecting, "drain inflight"},
		{StateTearingDown, "releasing resources"},
		{StateCompleted, "done"},
	}

	for _, tr := range transitions {
		if err := exp.TransitionTo(tr.state, tr.reason); err != nil {
			t.Fatalf("transition to %s failed: %v", tr.state, err)
		}
	}

	log := exp.GetStateLog()
	if len(log) != len(transitions) {
		t.Fatalf("log has %d entries, want %d", len(log), len(transitions))
	}

	// Verify each log entry matches.
	for i, entry := range log {
		if entry.To != transitions[i].state {
			t.Errorf("log[%d].To = %s, want %s", i, entry.To, transitions[i].state)
		}
		if entry.Reason != transitions[i].reason {
			t.Errorf("log[%d].Reason = %q, want %q", i, entry.Reason, transitions[i].reason)
		}
		if entry.Timestamp.IsZero() {
			t.Errorf("log[%d].Timestamp should not be zero", i)
		}
	}

	// Verify From→To chain is consistent.
	if log[0].From != StatePending {
		t.Errorf("first transition should be from PENDING, got %s", log[0].From)
	}
	for i := 1; i < len(log); i++ {
		if log[i].From != log[i-1].To {
			t.Errorf("log[%d].From=%s != log[%d].To=%s: broken chain",
				i, log[i].From, i-1, log[i-1].To)
		}
	}
}

// ============================================================================
// ES9: Failed Experiment Records Error Reason
// ============================================================================

func TestES9_FailedExperimentRecordsError(t *testing.T) {
	exp := NewExperiment("test-es9", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning)

	errorReason := "Slurm node crashed: slurmctld timeout after 180s"
	if err := exp.TransitionTo(StateFailed, errorReason); err != nil {
		t.Fatalf("transition to FAILED should succeed: %v", err)
	}

	if got := exp.GetError(); got != errorReason {
		t.Errorf("error = %q, want %q", got, errorReason)
	}
}

// ============================================================================
// ES10: Timestamp Ordering
// ============================================================================

func TestES10_TimestampOrdering(t *testing.T) {
	exp := NewExperiment("test-es10", ExperimentInput{})
	createdAt := exp.CreatedAt

	mustTransition(t, exp, StateProvisioning)

	exp.mu.RLock()
	updatedAt := exp.UpdatedAt
	exp.mu.RUnlock()

	if updatedAt.Before(createdAt) {
		t.Errorf("UpdatedAt (%v) should be >= CreatedAt (%v)", updatedAt, createdAt)
	}
}

// ============================================================================
// ES7: Concurrent State Transitions (Race Detector Verification)
// ============================================================================

func TestES7_ConcurrentTransitions(t *testing.T) {
	// This test must pass with `go test -race`.
	// Multiple goroutines attempt to fail the same experiment concurrently.
	// Exactly one should succeed; the rest should get TransitionError.
	exp := NewExperiment("test-es7", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning)

	const goroutines = 50
	var wg sync.WaitGroup
	successes := make(chan int, goroutines)

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(id int) {
			defer wg.Done()
			err := exp.TransitionTo(StateFailed, fmt.Sprintf("goroutine-%d", id))
			if err == nil {
				successes <- id
			}
		}(i)
	}
	wg.Wait()
	close(successes)

	count := 0
	for range successes {
		count++
	}
	if count != 1 {
		t.Errorf("expected exactly 1 successful transition, got %d", count)
	}
	if exp.GetState() != StateFailed {
		t.Errorf("state = %s, want FAILED", exp.GetState())
	}
}

func TestConcurrentReads(t *testing.T) {
	// Verify concurrent reads don't race with writes.
	exp := NewExperiment("test-concurrent-reads", ExperimentInput{})

	var wg sync.WaitGroup
	wg.Add(100)
	for i := 0; i < 100; i++ {
		go func() {
			defer wg.Done()
			_ = exp.GetState()
			_ = exp.GetStateLog()
			_ = exp.GetError()
		}()
	}

	// Simultaneously do a write.
	_ = exp.TransitionTo(StateProvisioning, "concurrent write")
	wg.Wait()
}

// ============================================================================
// Progress Tracking
// ============================================================================

func TestSetProgress(t *testing.T) {
	exp := NewExperiment("test-progress", ExperimentInput{})
	exp.SetProgress(3, 12)

	exp.mu.RLock()
	defer exp.mu.RUnlock()
	if exp.CellsCompleted != 3 {
		t.Errorf("CellsCompleted = %d, want 3", exp.CellsCompleted)
	}
	if exp.CellsTotal != 12 {
		t.Errorf("CellsTotal = %d, want 12", exp.CellsTotal)
	}
}

// ============================================================================
// IsValidTransition Truth Table (Exhaustive)
// ============================================================================

func TestIsValidTransition_TruthTable(t *testing.T) {
	// Test every (from, to) pair for the 10 valid states.
	allStates := []ExperimentState{
		StatePending, StateProvisioning, StateDeploying, StateWarmingUp,
		StateBenchmarking, StateCollecting, StateTearingDown,
		StateCompleted, StateFailed, StateCancelled,
	}

	for _, from := range allStates {
		for _, to := range allStates {
			result := IsValidTransition(from, to)
			t.Run(fmt.Sprintf("%s_to_%s", from, to), func(t *testing.T) {
				expected := expectedTransitionResult(from, to)
				if result != expected {
					t.Errorf("IsValidTransition(%s, %s) = %v, want %v",
						from, to, result, expected)
				}
			})
		}
	}
}

// expectedTransitionResult encodes the ground truth for the exhaustive test.
func expectedTransitionResult(from, to ExperimentState) bool {
	// Terminal states cannot transition anywhere.
	if from.IsTerminal() {
		return false
	}
	// Self-transitions are never valid.
	if from == to {
		return false
	}
	// Any non-terminal → FAILED or CANCELLED is valid.
	if to == StateFailed || to == StateCancelled {
		return true
	}
	// Only the strict next forward state is valid.
	next, ok := forwardTransitions[from]
	return ok && next == to
}

// ============================================================================
// Test Helpers
// ============================================================================

// mustTransition applies a sequence of transitions, failing the test on error.
func mustTransition(t *testing.T, exp *Experiment, states ...ExperimentState) {
	t.Helper()
	for _, s := range states {
		if err := exp.TransitionTo(s, "test setup"); err != nil {
			t.Fatalf("setup transition to %s failed: %v", s, err)
		}
	}
}

// assertTransitionError asserts that the error is a *TransitionError.
func assertTransitionError(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected TransitionError, got nil")
	}
	if _, ok := err.(*TransitionError); !ok {
		t.Fatalf("expected *TransitionError, got %T: %v", err, err)
	}
}

// completedExperiment creates an experiment that has traversed the full lifecycle.
func completedExperiment(t *testing.T) *Experiment {
	t.Helper()
	exp := NewExperiment("test-completed", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning, StateDeploying, StateWarmingUp,
		StateBenchmarking, StateCollecting, StateTearingDown, StateCompleted)
	return exp
}

// failedExperiment creates an experiment that failed during provisioning.
func failedExperiment(t *testing.T) *Experiment {
	t.Helper()
	exp := NewExperiment("test-failed", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning)
	if err := exp.TransitionTo(StateFailed, "GPU allocation timeout"); err != nil {
		t.Fatalf("transition to FAILED failed: %v", err)
	}
	return exp
}

// cancelledExperiment creates an experiment cancelled during benchmarking.
func cancelledExperiment(t *testing.T) *Experiment {
	t.Helper()
	exp := NewExperiment("test-cancelled", ExperimentInput{})
	mustTransition(t, exp, StateProvisioning, StateDeploying, StateWarmingUp, StateBenchmarking)
	cancelled, _ := exp.Cancel("user cancelled")
	if !cancelled {
		t.Fatal("cancellation should succeed")
	}
	return exp
}
