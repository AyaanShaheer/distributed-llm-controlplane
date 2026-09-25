package orchestrator

import (
	"crypto/rand"
	"fmt"
	"sync"
)

// ============================================================================
// Store Error Types
// ============================================================================

// ErrAlreadyExists is returned when attempting to create an experiment with a duplicate ID.
type ErrAlreadyExists struct {
	ExperimentID string
}

func (e *ErrAlreadyExists) Error() string {
	return fmt.Sprintf("experiment %q already exists", e.ExperimentID)
}

// ErrNotFound is returned when an experiment ID does not exist in the store.
type ErrNotFound struct {
	ExperimentID string
}

func (e *ErrNotFound) Error() string {
	return fmt.Sprintf("experiment %q not found", e.ExperimentID)
}

// ============================================================================
// Experiment Store
// ============================================================================

// ExperimentStore is a thread-safe in-memory registry of profiling experiments.
// It is the single authority for experiment creation, lookup, and enumeration.
//
// In a production deployment, this would be backed by a persistent store
// (e.g., etcd, PostgreSQL). The in-memory implementation is suitable for
// single-daemon deployments and testing.
type ExperimentStore struct {
	mu          sync.RWMutex
	experiments map[string]*Experiment
}

// NewExperimentStore creates an empty experiment store.
func NewExperimentStore() *ExperimentStore {
	return &ExperimentStore{
		experiments: make(map[string]*Experiment),
	}
}

// Create validates and registers a new experiment.
//
// If spec.ExperimentID is empty, a server-generated UUID is assigned.
// If spec.ExperimentID is provided and already exists, returns ErrAlreadyExists.
//
// The experiment starts in StatePending.
func (s *ExperimentStore) Create(spec ExperimentInput) (*Experiment, error) {
	// Validate the spec before accepting it.
	result := ValidateExperimentSpec(spec)
	if !result.IsValid() {
		return nil, fmt.Errorf("invalid experiment spec: %s", result.ErrorMessages())
	}

	id := spec.ExperimentID
	if id == "" {
		id = generateID()
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.experiments[id]; exists {
		return nil, &ErrAlreadyExists{ExperimentID: id}
	}

	exp := NewExperiment(id, spec)
	s.experiments[id] = exp
	return exp, nil
}

// Get retrieves an experiment by ID.
// Returns ErrNotFound if the ID does not exist.
func (s *ExperimentStore) Get(id string) (*Experiment, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	exp, exists := s.experiments[id]
	if !exists {
		return nil, &ErrNotFound{ExperimentID: id}
	}
	return exp, nil
}

// List returns all experiments in the store.
// Returns an empty slice (not nil) if no experiments exist.
func (s *ExperimentStore) List() []*Experiment {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]*Experiment, 0, len(s.experiments))
	for _, exp := range s.experiments {
		result = append(result, exp)
	}
	return result
}

// Count returns the number of experiments in the store.
func (s *ExperimentStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.experiments)
}

// ============================================================================
// ID Generation
// ============================================================================

// generateID produces a random 16-character hex string suitable for experiment IDs.
// Format: "exp-" prefix + 12 hex chars (e.g., "exp-a1b2c3d4e5f6").
func generateID() string {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		// Fallback to timestamp-based ID if crypto/rand fails (extremely unlikely).
		return fmt.Sprintf("exp-fallback-%d", uint64(0))
	}
	return fmt.Sprintf("exp-%x", b)
}
