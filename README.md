# Distributed LLM Inference Profiler & Orchestrator

An enterprise-grade control plane engineered to orchestrate, profile, and tune distributed LLM inference workloads (e.g. Llama-3) across multi-GPU clusters using **NVIDIA Triton Inference Server**, **TensorRT-LLM**, **vLLM**, and **Slurm**.

---

## Architecture Overview

```
+---------------------------------------------------------------------------------------------------+
|                                       CI/CD & Developer Client                                    |
|   (GitHub Actions Workflow / CLI Client / Automated Regression Gating Budget Check)               |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  | gRPC (RunExperiment / StreamTelemetry)
                                                  v
+---------------------------------------------------------------------------------------------------+
|                        Go Control Plane Orchestrator (Daemon / Core Service)                      |
|                                                                                                   |
|  +---------------------+   +---------------------+   +---------------------+   +---------------+  |
|  | Experiment Engine   |   | Matrix Sweep Engine |   | Traffic Generator   |   | Telemetry &   |  |
|  | & Job State Machine |-->| (TP, Batch, Quant)  |-->| (Poisson / Pareto)  |-->| Regression    |  |
|  +----------+----------+   +----------+----------+   +----------+----------+   | Evaluator     |  |
|             |                         |                         |              +-------+-------+  |
+-------------|-------------------------|-------------------------|----------------------|----------+
              | Cluster Allocation      | Model Config Generation | Streaming Inference  | Telemetry Query
              v                         v                         v                      v
+-----------------------------+ +-------------------------------+ +-------------------+ +--------------+
| HPC / Cluster Scheduler API | | Model Store & Config Engine   | | Triton Inference  | | Observability|
| - Slurm REST API / CLI      | | - Dynamic config.pbtxt        | | Server Clusters   | | - Prometheus |
| - sbatch / srun / scancel   | | - Engine build artifacts      | | (v2 gRPC / HTTP)  | | - NVIDIA DCGM|
| - Pyxis / Enroot Containers | | - S3 / Shared NFS / Lustre    | | - KV-Cache Paging | | - Node Exp.  |
+-----------------------------+ +-------------------------------+ +-------------------+ +--------------+
```

---

## Key Documentation

- **[High-Level Design (HLD)](./HIGH_LEVEL_DESIGN.md)**: Deep dive into control plane architecture, Triton C++ backend integration, and cluster scheduling.
- **[Back-of-the-Envelope Estimates](./BACK_OF_ENVELOPE_ESTIMATES.md)**: Mathematical derivations of KV-cache memory, roofline latency bounds, and NVLink interconnect overhead.
- **[Engineering Roadmap & Tech Stack](./ROADMAP.md)**: 6-phase implementation roadmap, system requirements, and sprint milestones.
- **[API Edge Cases & Invariants](./api/proto/v1/EDGE_CASES.md)**: 40 edge cases and validation rules covering specs, sweeps, traffic, and budgets.

---

## Tech Stack

| Domain | Technology |
| :--- | :--- |
| **Control Plane Core** | **Go (Golang 1.25+)** |
| **API & RPC** | **gRPC & Protocol Buffers (`proto3`)** |
| **Model Server** | **NVIDIA Triton Inference Server (24.04+)** |
| **Inference Engines** | **TensorRT-LLM** & **vLLM** |
| **HPC Workload Manager** | **Slurm** (with GRES GPU allocation & Pyxis/Enroot containers) |
| **Hardware Telemetry** | **NVIDIA DCGM** (Data Center GPU Manager) & Prometheus |
| **CI/CD Automation** | **GitHub Actions** (Performance regression budget enforcement) |

---

## Getting Started

### Prerequisites
- Go 1.22+ installed
- Git

### Running Tests
```bash
go test -v -count=1 ./...
```
*(All 52 edge-case validation tests pass)*
