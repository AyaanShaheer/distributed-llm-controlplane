# Engineering Roadmap & Technical Specifications

## 1. System Requirements & Hardware Targets

To enable development and validation both on enterprise multi-GPU clusters and on personal developer machines, the project defines two execution profiles: **Production HPC Profile** and **Local / Emulation Profile**.

### 1.1. Production HPC Cluster Profile (Target Deployment)
- **Compute**: Minimum 2 to 8 NVIDIA GPUs interconnected via NVLink (e.g., $2\times$ or $4\times$ or $8\times$ A100-80GB / H100-80GB / H200 / L40S, or dual RTX 3090/4090 with PCIe P2P).
  - *Cloud On-Demand options*: RunPod, Lambda Labs, Nebius AI, or CoreWeave.
- **Interconnect**: NVLink (intra-node) and InfiniBand HDR/NDR (200/400 Gbps, inter-node).
- **Cluster Scheduler**: Slurm workload manager (v22.05+ or v23.02+) with GRES GPU allocation plugin and Pyxis/Enroot container plugin.
- **Host OS & Drivers**: Ubuntu 22.04 LTS, NVIDIA Driver $\ge 535.129.03$ (CUDA 12.2+).
- **Shared Storage**: High-throughput shared filesystem (NFS, Lustre, or GPFS) for model weights and engine repositories.

### 1.2. Local Emulation Profile (Developer & CI/CD Sandbox)
No multi-GPU server is required to develop, test, or demonstrate the entire control plane:
- **Workstation / Laptop**: Windows 11 with WSL2 (Ubuntu 22.04) or native Linux/macOS.
- **Local Accelerator**: Any single NVIDIA GPU (e.g., RTX 3060/4070/4090) OR pure CPU mode.
- **Containerized Slurm Mock**: Multi-container Docker Compose cluster running a virtual Slurm master (`slurmctld`), worker nodes (`slurmd`), and a mock Triton/vLLM daemon with synthetic response streaming.
- **Tooling**: Docker Desktop / Podman, Go 1.22+, `protoc` compiler, `grpcurl`.

---

## 2. Complete Technical Stack

| Tier / Domain | Technology | Rationale & NVIDIA Alignment |
| :--- | :--- | :--- |
| **Control Plane Core** | **Go (Golang 1.22+)** | Memory-efficient concurrency (goroutines, channels), lightweight binaries, native Docker/K8s/Slurm API client ecosystem. |
| **RPC & Interface** | **gRPC & Protocol Buffers (proto3)** | High-throughput bi-directional streaming, strict schema definition, matches Triton v2 gRPC client standards. |
| **Serving Runtime** | **NVIDIA Triton Inference Server (24.04+)** | NVIDIA's flagship multi-backend model server; dynamic batching, model management, C++ execution pipelines. |
| **Inference Engines** | **TensorRT-LLM** & **vLLM** | TensorRT-LLM for state-of-the-art C++ kernel execution on Tensor Cores; vLLM for baseline comparison and rapid local iteration. |
| **HPC Workload Manager**| **Slurm Workload Manager** | Industry-standard HPC batch scheduler used in NVIDIA DGX SuperPODs; provides GRES GPU topology allocation. |
| **Container Runtime** | **NVIDIA Container Toolkit & Pyxis/Enroot**| Allows containerized Triton workloads to execute directly inside Slurm jobs with zero user-privilege elevation. |
| **Hardware Telemetry** | **NVIDIA DCGM (Data Center GPU Manager)** | Enterprise hardware telemetry: SM active %, Tensor Core utilization, memory bandwidth, thermal throttle tracking. |
| **Serving Telemetry** | **Prometheus & OpenTelemetry** | Aggregates Triton `:8002/metrics` and custom Go client-side latency histograms (TTFT, ITL, QPS). |
| **Dashboard & Viz** | **Grafana & Static HTML/SVG Exporters** | Live monitoring dashboards during profiling runs and automatic static report generation for pull requests. |
| **CI/CD Automation** | **GitHub Actions Runner** | Contract regression testing on PRs; parses benchmark outputs and enforces latency budgets. |

---

## 3. Phased Implementation Roadmap

```
+---------------------------------------------------------------------------------------+
| PHASE 1: Protocol & Architectural Scaffolding                                         |
|  - Define Protobuf schemas (ExperimentSpec, Metrics, JobStatus)                      |
|  - Implement Go gRPC daemon skeleton and CLI client                                   |
|  - Build Dockerized Virtual Slurm + Mock Triton cluster for local dev                 |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| PHASE 2: Triton & Slurm Orchestration Adapters                                        |
|  - Slurm job generator (sbatch/srun script templating with GRES gpu:N)                |
|  - Pyxis container launch automation with NVLink passthrough                          |
|  - Triton dynamic config.pbtxt generator (TP, in-flight batching, KV allocation)      |
|  - Triton v2 gRPC health probe & model ready lifecycle state machine                  |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| PHASE 3: High-Concurrency Load Generator & Precision Profiler                         |
|  - Zero-allocation Go gRPC streaming client for Triton v2 API                         |
|  - Nanosecond timestamping for TTFT (chunk 0) and ITL (inter-chunk intervals)         |
|  - Synthetic prompt distributions (Poisson arrival, Pareto sequence lengths)          |
|  - Batch sweep driver (matrix sweep across batch sizes 1..128 and TP 1..8)             |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| PHASE 4: Telemetry Aggregation & Roofline Profiling                                   |
|  - Integration with NVIDIA DCGM Exporter and Prometheus scrapers                      |
|  - Compute arithmetic intensity, MFU (Model FLOPs Utilization), and memory bandwidth  |
|  - Detect CUDA OOM and out-of-bounds parameter sets gracefully                        |
|  - Pareto Frontier calculator: identifies optimal TP vs Batch Size tradeoffs          |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| PHASE 5: CI/CD Performance Regression Gating                                          |
|  - GitHub Actions automated benchmark action                                          |
|  - Budget enforcement engine (`budget.yaml`) verifying p99 TTFT and p95 ITL           |
|  - Automated PR markdown comment generator with latency graphs & throughput charts    |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| PHASE 6: Production Validation & Portfolio Polish                                     |
|  - Run full matrix benchmarks on real GPU hardware (RunPod / Lambda / DGX)            |
|  - Publish whitepaper / technical report comparing TRT-LLM vs vLLM on Llama-3-8B      |
|  - Interactive terminal dashboard (TUI) and video demonstration                       |
+---------------------------------------------------------------------------------------+
```

---

## 4. Detailed Sprint Breakdown

### Milestone 1: Protocols, Scaffolding & Virtual Cluster (Week 1–2)
- [ ] Initialize Go workspace (`go.mod`) with clean package structure (`cmd/`, `internal/`, `api/`).
- [ ] Author `api/proto/v1/orchestrator.proto`:
  - Enums: `ServingBackend` (`TRT_LLM`, `VLLM`), `Quantization` (`FP16`, `FP8`, `INT4`), `JobState`.
  - Messages: `ExperimentSpec`, `SweepDimensions`, `MetricSample`, `BenchmarkResult`.
- [ ] Compile Go gRPC code via `protoc-gen-go` and `protoc-gen-go-grpc`.
- [ ] Construct `deployments/docker/docker-compose.mock.yml` running:
  - 1x Mock Slurm Controller.
  - 2x Mock Slurm Compute Nodes.
  - 1x Mock Triton Server (simulating token streaming responses with realistic synthetic delays).
  - 1x Prometheus instance.

### Milestone 2: Scheduler Engine & Triton Integration (Week 3–4)
- [ ] Implement `internal/scheduler/slurm.go`:
  - Slurm CLI executor (`sbatch`, `squeue`, `scancel`) and Slurm REST API client.
  - Parameterized template renderer for `#SBATCH --nodes=N --gres=gpu:M --ntasks=M`.
  - Automatic port forwarding / discovery of assigned node IP and port.
- [ ] Implement `internal/triton/config_generator.go`:
  - Generates valid `config.pbtxt` files for TensorRT-LLM backend:
    - Setting `tensor_parallel_size`, `pipeline_parallel_size`.
    - Configuring `dynamic_batching`, `max_queue_delay_microseconds`.
    - Setting `kv_cache_free_gpu_mem_fraction`.
- [ ] Implement asynchronous polling for `/v2/health/ready` with configurable readiness timeouts.

### Milestone 3: Asynchronous Profiler & Traffic Injector (Week 5–6)
- [ ] Implement `internal/traffic/generator.go`:
  - High-throughput streaming gRPC client against Triton `InferenceServerClient`.
  - Configurable worker pools with dynamic rate limiters (Token Bucket and Leaky Bucket).
  - Synthetic workload generators:
    - Fixed length (e.g., 512 in / 128 out).
    - ShareGPT / LMSYS trace replay distribution.
    - Poisson arrival processes.
- [ ] Implement `internal/collector/latency_recorder.go`:
  - Nanosecond timer recording:
    - $T_{\text{submit}} \to T_{\text{first\_token}} \implies$ **TTFT**.
    - $T_{token\_i} \to T_{token\_i+1} \implies$ **ITL**.
  - Quantile calculation ($p50, p90, p95, p99$) using HdrHistogram or streaming t-digest.

### Milestone 4: Telemetry Aggregation & Pareto Optimization (Week 7–8)
- [ ] Implement `internal/collector/dcgm.go`:
  - Prometheus query client fetching instantaneous and average SM utilization, memory bandwidth, and power.
- [ ] Implement `internal/orchestrator/sweep.go`:
  - Grid sweep controller executing ordered test matrix ($TP \in [1, 2, 4]$, Batch $\in [1, 4, 16, 64]$).
  - Graceful crash recovery: records CUDA OOM without halting sweep pipeline.
- [ ] Pareto Frontier algorithm:
  - Computes the optimal Pareto curve mapping Cost/Throughput vs. p99 Tail Latency.
  - Outputs recommended optimal serving configuration for given SLA targets.

### Milestone 5: CI/CD Pipeline & GitHub Action (Week 9–10)
- [ ] Create `deployments/github-actions/action.yml`:
  - Action that runs on pull requests modifying `models/**` or `configs/**`.
  - Starts local or remote runner, triggers benchmark run, parses JSON results.
- [ ] Implement `internal/regression/evaluator.go`:
  - Evaluates delta against `main` branch baseline.
  - Fails CI build if $p99 \ \text{TTFT}$ degrades by $> 5\%$ or throughput drops by $> 5\%$.
  - Generates rich GitHub Markdown summary with comparison tables.

### Milestone 6: Real Hardware Benchmark & Portfolio Package (Week 11–12)
- [ ] Deploy to real multi-GPU instance (e.g., 2x or 4x A100/H100 on RunPod/Nebius).
- [ ] Execute comparative study: **TensorRT-LLM vs. vLLM on Llama-3-8B** under identical prompt traces.
- [ ] Write up comprehensive benchmarking whitepaper (`docs/BENCHMARK_REPORT.md`).
- [ ] Record a 3-minute terminal/architecture walkthrough video for hiring managers.

---

## 5. What Makes This Stand Out for NVIDIA DSX Enablement

1. **Native NVIDIA Ecosystem Depth**: You aren't just calling a Python script; you are programmatically generating Triton `config.pbtxt`, configuring the TensorRT-LLM C++ runtime, orchestrating MPI ranks, and parsing DCGM metrics.
2. **True HPC Scheduling**: You demonstrate first-hand fluency with Slurm, GRES GPU topologies, Pyxis/Enroot containers, and NVLink bandwidth limits.
3. **Rigorous Low-Level Systems Engineering**: Built in Go with zero-allocation streaming gRPC routines, robust error handling, and production-grade state machines.
4. **Pragmatic Production CI/CD**: Solves a real-world enterprise problem: preventing performance regressions in production LLM deployments before code merges.
