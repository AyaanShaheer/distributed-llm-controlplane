# Edge Cases & Invariants: Experiment State Machine

The state machine governs the entire lifecycle of a profiling experiment.
Every case listed here maps to a unit test in `state_machine_test.go` and `experiment_test.go`.

---

## State Transition Diagram

```
                    ┌──────────────────────────────────────────────┐
                    │              TERMINAL STATES                 │
                    │  ┌──────────┐ ┌────────┐ ┌───────────┐     │
                    │  │COMPLETED │ │ FAILED │ │ CANCELLED │     │
                    │  └──────────┘ └────────┘ └───────────┘     │
                    └──────────────────────────────────────────────┘
                          ▲              ▲            ▲
                          │              │            │
  PENDING ──► PROVISIONING ──► DEPLOYING ──► WARMING_UP ──► BENCHMARKING
                                                                   │
                                                                   ▼
                                              COMPLETED ◄── TEARING_DOWN ◄── COLLECTING
```

- Forward transitions follow the strict order above (no skipping).
- Any non-terminal state may transition to FAILED or CANCELLED.
- Terminal states (COMPLETED, FAILED, CANCELLED) are irreversible.

---

## State Transition Edge Cases (SM-*)

| # | Test Case | From State | To State | Expected Result |
|---|-----------|------------|----------|-----------------|
| SM1 | Valid forward: PENDING → PROVISIONING | PENDING | PROVISIONING | Accept |
| SM2 | Valid forward: PROVISIONING → DEPLOYING | PROVISIONING | DEPLOYING | Accept |
| SM3 | Valid forward: DEPLOYING → WARMING_UP | DEPLOYING | WARMING_UP | Accept |
| SM4 | Valid forward: WARMING_UP → BENCHMARKING | WARMING_UP | BENCHMARKING | Accept |
| SM5 | Valid forward: BENCHMARKING → COLLECTING | BENCHMARKING | COLLECTING | Accept |
| SM6 | Valid forward: COLLECTING → TEARING_DOWN | COLLECTING | TEARING_DOWN | Accept |
| SM7 | Valid forward: TEARING_DOWN → COMPLETED | TEARING_DOWN | COMPLETED | Accept |
| SM8 | Full lifecycle traversal (all 7 forward transitions) | PENDING | COMPLETED | Accept |
| SM9 | Skip states: PENDING → BENCHMARKING (E38) | PENDING | BENCHMARKING | Reject: invalid transition |
| SM10 | Skip states: PENDING → DEPLOYING | PENDING | DEPLOYING | Reject: invalid transition |
| SM11 | Skip states: PROVISIONING → WARMING_UP | PROVISIONING | WARMING_UP | Reject: invalid transition |
| SM12 | Backward: BENCHMARKING → WARMING_UP | BENCHMARKING | WARMING_UP | Reject: invalid transition |
| SM13 | Backward: DEPLOYING → PROVISIONING | DEPLOYING | PROVISIONING | Reject: invalid transition |
| SM14 | Self-transition: PENDING → PENDING | PENDING | PENDING | Reject: no-op self-transition |
| SM15 | Self-transition: BENCHMARKING → BENCHMARKING | BENCHMARKING | BENCHMARKING | Reject: no-op self-transition |

---

## Terminal State Invariants (SM-T*)

| # | Test Case | From State | To State | Expected Result |
|---|-----------|------------|----------|-----------------|
| SM-T1 | COMPLETED → BENCHMARKING (E36) | COMPLETED | BENCHMARKING | Reject: terminal state is irreversible |
| SM-T2 | COMPLETED → FAILED | COMPLETED | FAILED | Reject: terminal state is irreversible |
| SM-T3 | COMPLETED → CANCELLED | COMPLETED | CANCELLED | Reject: terminal state is irreversible |
| SM-T4 | COMPLETED → COMPLETED | COMPLETED | COMPLETED | Reject: terminal state is irreversible |
| SM-T5 | FAILED → PENDING (E37) | FAILED | PENDING | Reject: terminal state is irreversible |
| SM-T6 | FAILED → any forward state | FAILED | PROVISIONING | Reject: terminal state is irreversible |
| SM-T7 | FAILED → FAILED | FAILED | FAILED | Reject: terminal state is irreversible |
| SM-T8 | CANCELLED → PENDING | CANCELLED | PENDING | Reject: terminal state is irreversible |
| SM-T9 | CANCELLED → CANCELLED | CANCELLED | CANCELLED | Reject: terminal state is irreversible |

---

## Failure Transitions (SM-F*)

| # | Test Case | From State | To State | Expected Result |
|---|-----------|------------|----------|-----------------|
| SM-F1 | PENDING → FAILED | PENDING | FAILED | Accept |
| SM-F2 | PROVISIONING → FAILED | PROVISIONING | FAILED | Accept |
| SM-F3 | DEPLOYING → FAILED | DEPLOYING | FAILED | Accept |
| SM-F4 | WARMING_UP → FAILED | WARMING_UP | FAILED | Accept |
| SM-F5 | BENCHMARKING → FAILED | BENCHMARKING | FAILED | Accept |
| SM-F6 | COLLECTING → FAILED | COLLECTING | FAILED | Accept |
| SM-F7 | TEARING_DOWN → FAILED | TEARING_DOWN | FAILED | Accept |

---

## Cancellation Transitions (SM-C*)

| # | Test Case | From State | To State | Expected Result |
|---|-----------|------------|----------|-----------------|
| SM-C1 | PENDING → CANCELLED | PENDING | CANCELLED | Accept |
| SM-C2 | PROVISIONING → CANCELLED | PROVISIONING | CANCELLED | Accept |
| SM-C3 | BENCHMARKING → CANCELLED | BENCHMARKING | CANCELLED | Accept |
| SM-C4 | TEARING_DOWN → CANCELLED | TEARING_DOWN | CANCELLED | Accept |
| SM-C5 | Cancel already-completed experiment (E40) | COMPLETED | CANCELLED | Reject but NOT error; return warning |

---

## Experiment Store Edge Cases (ES-*)

| # | Test Case | Expected Result |
|---|-----------|-----------------|
| ES1 | Create experiment with server-generated ID | ID is non-empty UUID format |
| ES2 | Create experiment with user-provided ID | Uses user's ID exactly |
| ES3 | Duplicate experiment ID submission (E39) | Reject: ALREADY_EXISTS |
| ES4 | Get non-existent experiment by ID | Return NOT_FOUND error |
| ES5 | List experiments on empty store | Return empty slice, no error |
| ES6 | List experiments with multiple items | Return all experiments |
| ES7 | Concurrent state transitions from multiple goroutines | No data race (must pass -race) |
| ES8 | Transition audit log records every state change | StateLog contains all transitions in order |
| ES9 | Failed experiment records error reason | Error field is populated |
| ES10 | Experiment tracks wall-clock timestamps | CreatedAt < UpdatedAt after transitions |
