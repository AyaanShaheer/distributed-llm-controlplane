// Package orchestrator implements the core experiment lifecycle engine.
//
// The state machine enforces strictly ordered, irreversible state transitions
// for profiling experiments. It is the single source of truth for whether a
// transition is legal. All state mutations flow through TransitionTo(), which
// guarantees thread safety via a per-experiment mutex.
package orchestrator

import (
	"fmt"
	"sync"
	"time"
)

// ============================================================================
// State Type & Constants
// ============================================================================

// ExperimentState represents a discrete phase in the experiment lifecycle.
// Values intentionally mirror the protobuf ExperimentState enum.
type ExperimentState int32

const (
	StateUnspecified ExperimentState = 0
	StatePending     ExperimentState = 1
	StateProvisioning ExperimentState = 2
	StateDeploying   ExperimentState = 3
	StateWarmingUp   ExperimentState = 4
	StateBenchmarking ExperimentState = 5
	StateCollecting  ExperimentState = 6
	StateTearingDown ExperimentState = 7
	StateCompleted   ExperimentState = 8
	StateFailed      ExperimentState = 9
	StateCancelled   ExperimentState = 10
)

// stateNames provides human-readable names for logging and error messages.
var stateNames = map[ExperimentState]string{
	StateUnspecified:  "UNSPECIFIED",
	StatePending:      "PENDING",
	StateProvisioning: "PROVISIONING",
	StateDeploying:    "DEPLOYING",
	StateWarmingUp:    "WARMING_UP",
	StateBenchmarking: "BENCHMARKING",
	StateCollecting:   "COLLECTING",
	StateTearingDown:  "TEARING_DOWN",
	StateCompleted:    "COMPLETED",
	StateFailed:       "FAILED",
	StateCancelled:    "CANCELLED",
}

// String returns the human-readable name of the state.
func (s ExperimentState) String() string {
	if name, ok := stateNames[s]; ok {
		return name
	}
	return fmt.Sprintf("UNKNOWN(%d)", s)
}

// ============================================================================
// Transition Rules
// ============================================================================

// terminalStates are states from which no further transitions are allowed.
var terminalStates = map[ExperimentState]bool{
	StateCompleted: true,
	StateFailed:    true,
	StateCancelled: true,
}

// forwardTransitions defines the only valid "happy path" progression.
// Key = current state, Value = the single next valid forward state.
var forwardTransitions = map[ExperimentState]ExperimentState{
	StatePending:      StateProvisioning,
	StateProvisioning: StateDeploying,
	StateDeploying:    StateWarmingUp,
	StateWarmingUp:    StateBenchmarking,
	StateBenchmarking: StateCollecting,
	StateCollecting:   StateTearingDown,
	StateTearingDown:  StateCompleted,
}

// IsTerminal returns true if the state is a terminal (irreversible) state.
func (s ExperimentState) IsTerminal() bool {
	return terminalStates[s]
}

// IsValidTransition checks whether transitioning from `from` to `to` is legal.
//
// Rules:
//  1. Terminal states cannot transition to anything (including themselves).
//  2. Self-transitions are never allowed (no-op guard).
//  3. Any non-terminal state may transition to FAILED or CANCELLED.
//  4. Forward transitions must follow the strict ordered sequence.
func IsValidTransition(from, to ExperimentState) bool {
	// Rule 1: Terminal states are absorbing.
	if from.IsTerminal() {
		return false
	}

	// Rule 2: Self-transitions are disallowed.
	if from == to {
		return false
	}

	// Rule 3: Any non-terminal state can fail or be cancelled.
	if to == StateFailed || to == StateCancelled {
		return true
	}

	// Rule 4: Check strict forward progression.
	nextValid, exists := forwardTransitions[from]
	return exists && nextValid == to
}

// ============================================================================
// Transition Error Types
// ============================================================================

// TransitionError is returned when a state transition is rejected.
type TransitionError struct {
	ExperimentID string
	From         ExperimentState
	To           ExperimentState
	Reason       string
}

func (e *TransitionError) Error() string {
	return fmt.Sprintf("experiment %q: invalid transition %s → %s: %s",
		e.ExperimentID, e.From, e.To, e.Reason)
}

// ============================================================================
// State Transition Record (Audit Log)
// ============================================================================

// StateTransition records a single state change for audit and debugging.
type StateTransition struct {
	From      ExperimentState
	To        ExperimentState
	Timestamp time.Time
	Reason    string // Why the transition occurred (e.g., "GPU allocation complete").
}

// ============================================================================
// Experiment — The Core Stateful Entity
// ============================================================================

// Experiment holds the full lifecycle state of a single profiling experiment.
// All state mutations are serialized through the embedded mutex.
type Experiment struct {
	mu sync.RWMutex

	// Identity & configuration.
	ID        string
	Spec      ExperimentInput
	CreatedAt time.Time
	UpdatedAt time.Time

	// Lifecycle state.
	State    ExperimentState
	StateLog []StateTransition

	// Failure context (populated only when State == StateFailed).
	Error string

	// Progress tracking.
	CellsCompleted uint32
	CellsTotal     uint32
}

// NewExperiment creates a new experiment in the PENDING state.
func NewExperiment(id string, spec ExperimentInput) *Experiment {
	now := time.Now()
	return &Experiment{
		ID:        id,
		Spec:      spec,
		State:     StatePending,
		CreatedAt: now,
		UpdatedAt: now,
		StateLog:  make([]StateTransition, 0, 8),
	}
}

// GetState returns the current state (thread-safe read).
func (e *Experiment) GetState() ExperimentState {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.State
}

// GetStateLog returns a copy of the transition audit log (thread-safe).
func (e *Experiment) GetStateLog() []StateTransition {
	e.mu.RLock()
	defer e.mu.RUnlock()
	log := make([]StateTransition, len(e.StateLog))
	copy(log, e.StateLog)
	return log
}

// GetError returns the error reason if the experiment has failed.
func (e *Experiment) GetError() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.Error
}

// TransitionTo attempts to move the experiment to the target state.
// Returns nil on success, or a *TransitionError explaining the rejection.
//
// The `reason` parameter is recorded in the audit log for debugging.
// If transitioning to StateFailed, `reason` is also stored in the Error field.
func (e *Experiment) TransitionTo(target ExperimentState, reason string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	current := e.State

	if !IsValidTransition(current, target) {
		errReason := "invalid state transition"
		if current.IsTerminal() {
			errReason = fmt.Sprintf("terminal state %s is irreversible", current)
		} else if current == target {
			errReason = "self-transitions are not allowed"
		} else {
			expected, ok := forwardTransitions[current]
			if ok {
				errReason = fmt.Sprintf("expected next state is %s, not %s", expected, target)
			}
		}
		return &TransitionError{
			ExperimentID: e.ID,
			From:         current,
			To:           target,
			Reason:       errReason,
		}
	}

	now := time.Now()
	e.StateLog = append(e.StateLog, StateTransition{
		From:      current,
		To:        target,
		Timestamp: now,
		Reason:    reason,
	})
	e.State = target
	e.UpdatedAt = now

	// If transitioning to FAILED, persist the reason.
	if target == StateFailed {
		e.Error = reason
	}

	return nil
}

// Cancel attempts graceful cancellation of the experiment.
// Returns (true, nil) if cancelled, (false, nil) if already terminal (no-op warning),
// or (false, error) on unexpected failure.
func (e *Experiment) Cancel(reason string) (cancelled bool, warning string) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.State.IsTerminal() {
		return false, fmt.Sprintf("experiment %q is already in terminal state %s; cancel is a no-op",
			e.ID, e.State)
	}

	now := time.Now()
	e.StateLog = append(e.StateLog, StateTransition{
		From:      e.State,
		To:        StateCancelled,
		Timestamp: now,
		Reason:    reason,
	})
	e.State = StateCancelled
	e.UpdatedAt = now
	return true, ""
}

// SetProgress updates the cell completion counters (thread-safe).
func (e *Experiment) SetProgress(completed, total uint32) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.CellsCompleted = completed
	e.CellsTotal = total
	e.UpdatedAt = time.Now()
}
