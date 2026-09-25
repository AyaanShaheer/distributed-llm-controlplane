# Back-of-the-Envelope Estimates: LLM Inference, Memory, & Profiling

This document outlines the first-principles mathematical formulas, memory sizing, compute roofs, and communication bottlenecks governing distributed LLM serving on modern NVIDIA GPU architectures (Ampere, Ada Lovelace, Hopper, and Blackwell).

---

## 1. Model Memory Sizing & KV-Cache Sizing

### 1.1. Model Weight Memory
The static GPU memory footprint required to hold model parameters:
$$\text{Memory}_{\text{weights}} = \text{Parameters} \times \text{Bytes per Parameter}$$

| Model | Parameters ($N$) | Precision | Bytes / Param | Raw Weight Memory | Minimum GPUs (24GB) | Minimum GPUs (80GB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama-3-8B** | $8.03 \times 10^9$ | FP16 / BF16 | 2 bytes | **~16.06 GB** | 1x (e.g. RTX 3090/4090) | 1x (A100 / H100) |
| **Llama-3-8B** | $8.03 \times 10^9$ | FP8 (E4M3) | 1 byte | **~8.03 GB** | 1x | 1x |
| **Llama-3-8B** | $8.03 \times 10^9$ | INT4 (AWQ/GPTQ)| 0.5 bytes | **~4.02 GB** | 1x | 1x |
| **Llama-3-70B**| $70.6 \times 10^9$ | FP16 / BF16 | 2 bytes | **~141.2 GB** | 8x (TP=8, 19GB/GPU) | 2x (TP=2, 71GB/GPU) |
| **Llama-3-70B**| $70.6 \times 10^9$ | FP8 (E4M3) | 1 byte | **~70.6 GB** | 4x (TP=4, 18GB/GPU) | 1x (TP=1, 71GB/GPU) |
| **Llama-3-70B**| $70.6 \times 10^9$ | INT4 (AWQ/GPTQ)| 0.5 bytes | **~35.3 GB** | 2x (TP=2, 18GB/GPU) | 1x (TP=1, 36GB/GPU) |

*Note: Runtime memory also requires ~15-20% margin for CUDA context, runtime activation buffers, and NCCL communicators (approx. 1.2–2.0 GB per GPU).*

---

### 1.2. KV-Cache Memory Calculation
For auto-regressive transformer models utilizing Grouped-Query Attention (GQA), the KV cache stores key and value tensors for every token in the context window across all transformer layers.

#### Formula:
$$\text{KV Cache Bytes per Token} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{Bytes per Element}$$
$$\text{Total KV Cache} = \text{KV Cache Bytes per Token} \times \text{Sequence Length} \times \text{Batch Size}$$

#### Case Study: Llama-3-8B
- Layers ($n_{\text{layers}}$): $32$
- Query Heads: $32$
- KV Heads ($n_{\text{kv\_heads}}$): $8$ (Grouped Query Attention, 4:1 ratio)
- Head Dimension ($d_{\text{head}}$): $128$
- Data Type: FP16 ($2$ bytes) or FP8 ($1$ byte)

$$\text{Bytes per Token (FP16)} = 2 \times 32 \times 8 \times 128 \times 2 = 131,072 \text{ bytes} \approx 128 \text{ KB / token}$$
$$\text{Bytes per Token (FP8)} = 2 \times 32 \times 8 \times 128 \times 1 = 65,536 \text{ bytes} \approx 64 \text{ KB / token}$$

#### Concurrency Limits on Single 80GB H100 (Llama-3-8B FP16):
- Total HBM3: $80 \text{ GB} = 81,920 \text{ MB}$
- Model Weights: $16,060 \text{ MB}$
- CUDA / Runtime Buffers: $2,000 \text{ MB}$
- Remaining for Paged KV-Cache: $81,920 - 18,060 \approx 63,860 \text{ MB}$ ($63.86 \text{ GB}$)
- At **4,096 tokens per request** (prompt + completion):
  $$\text{KV Memory per Request} = 4,096 \times 128 \text{ KB} = 524,288 \text{ KB} = 512 \text{ MB}$$
  $$\text{Max Concurrent Active Requests} = \frac{63,860 \text{ MB}}{512 \text{ MB}} \approx \mathbf{124 \text{ concurrent streams}}$$
- If quantized to **FP8 KV-Cache**:
  $$\text{Max Concurrent Active Requests} = \frac{63,860 \text{ MB}}{256 \text{ MB}} \approx \mathbf{249 \text{ concurrent streams}}$$

---

## 2. Compute Roofline & Latency Bounds

LLM inference consists of two fundamentally distinct operational regimes:
1. **Prefill Phase (Prompt Processing)**: Compute-bound (high arithmetic intensity, Matrix-Matrix multiplication / GEMM).
2. **Decode Phase (Token Generation)**: Memory Bandwidth-bound (low arithmetic intensity, Matrix-Vector multiplication / GEMV).

### 2.1. Theoretical Minimum Decode Latency (Batch Size = 1)
During auto-regressive decoding with batch size = 1, generating 1 token requires reading **all model weights from HBM to SRAM** exactly once:
$$\text{Arithmetic Intensity} = \frac{2 \times \text{Parameters}}{\text{Model Bytes}} = \frac{2 \text{ FLOPs}}{2 \text{ Bytes}} = 1 \text{ FLOP/Byte}$$
Because modern GPUs deliver thousands of FLOPs per byte of memory bandwidth, the decode phase is completely throttled by memory bandwidth ($BW_{\text{mem}}$):
$$\text{ITL}_{\text{min}} = \frac{\text{Model Weights Size (Bytes)}}{\text{GPU Memory Bandwidth (Bytes/s)}}$$

| GPU Target | Memory Bandwidth ($BW_{\text{mem}}$) | Llama-3-8B FP16 (16 GB) $\text{ITL}_{\text{min}}$ | Max Theoretical Tokens/sec ($B=1$) |
| :--- | :--- | :--- | :--- |
| **RTX 4090** | 1.008 TB/s | $16 / 1008 \approx 15.87 \text{ ms}$ | **~63 tokens/sec** |
| **A100 SXM4 (80GB)**| 2.039 TB/s | $16 / 2039 \approx 7.84 \text{ ms}$ | **~127 tokens/sec** |
| **H100 SXM5 (80GB)**| 3.350 TB/s | $16 / 3350 \approx 4.77 \text{ ms}$ | **~209 tokens/sec** |
| **H200 SXM5 (141GB)**| 4.800 TB/s | $16 / 4800 \approx 3.33 \text{ ms}$ | **~300 tokens/sec** |

*Key Takeaway for Profiler*: At small batch sizes, throughput is strictly bandwidth limited. As batch size increases, weights are reused across $B$ queries, increasing arithmetic intensity until hitting the Tensor Core compute roof.

---

### 2.2. Prefill Phase Throughput & TTFT
For prompt length $S_{\text{in}}$ and model parameters $P$:
$$\text{FLOPs}_{\text{prefill}} \approx 2 \times P \times S_{\text{in}}$$
On an H100 SXM (delivering ~989 TFLOPS FP16 Tensor Core, or ~1,979 TFLOPS FP8):
For a 2,048 token prompt on Llama-3-8B:
$$\text{FLOPs} = 2 \times 8.03 \times 10^9 \times 2,048 \approx 32.89 \times 10^{12} \text{ FLOPs} = 32.89 \text{ TFLOPs}$$
Assuming realistic 50% MFU (Model FLOPs Utilization):
$$\text{Effective Throughput} = 0.50 \times 989 \text{ TFLOPS} = 494.5 \text{ TFLOPS}$$
$$\text{Theoretical Compute TTFT} = \frac{32.89 \text{ TFLOPs}}{494.5 \text{ TFLOPS}} \approx \mathbf{66.5 \text{ ms}}$$
Plus KV-cache allocation and attention kernel launch overheads $\implies$ **Expected Real-World TTFT: ~75–95 ms**.

---

## 3. Distributed Interconnect & Communication Latency

When scaling to Tensor Parallelism ($TP$) across multiple GPUs, each transformer layer requires **2 All-Reduce operations** (one in the Attention block, one in the MLP block):
$$\text{All-Reduce Data Volume per Layer} = 2 \times \left( \frac{TP - 1}{TP} \right) \times B \times d_{\text{model}} \times \text{Bytes per Element}$$
$$\text{Total Inter-GPU Traffic per Token} = n_{\text{layers}} \times 2 \times \text{Data Volume per Layer}$$

#### For Llama-3-8B ($d_{\text{model}} = 4096$, $n_{\text{layers}} = 32$, FP16, $TP = 2$):
$$\text{Data Volume per Layer} = 2 \times \left(\frac{1}{2}\right) \times 1 \times 4096 \times 2 = 8,192 \text{ bytes} = 8 \text{ KB}$$
$$\text{Total Data per Token} = 32 \times 2 \times 8 \text{ KB} = 512 \text{ KB / token}$$

#### Interconnect Latency Analysis:
1. **NVLink 4 (Hopper - 900 GB/s bidirectional)**:
   $$\text{Transfer Time} = \frac{512 \text{ KB}}{900 \text{ GB/s}} \approx 0.57 \ \mu\text{s}$$
   *Result*: Communication overhead is negligible (<0.01% of total step time).
2. **PCIe Gen4 x16 (64 GB/s theoretical, ~28 GB/s effective)**:
   $$\text{Transfer Time} = \frac{512 \text{ KB}}{28 \text{ GB/s}} \approx 18.2 \ \mu\text{s}$$
   Plus NCCL synchronization kernel launch latency (~5–10 $\mu$s per layer $\times$ 64 calls = ~400 $\mu$s).
   *Result*: Noticeable tail jitter.
3. **InfiniBand / Ethernet without NVLink across Nodes (100 Gbps = 12.5 GB/s)**:
   $$\text{Transfer Time} \ge 40.9 \ \mu\text{s} + \text{network switch hops} \ (>1-2 \text{ ms})$$
   *Crucial Architectural Rule*: **Never run Tensor Parallelism across nodes without NVLink/NVSwitch**. Multi-node configurations should use Pipeline Parallelism (PP) or Tensor Parallelism restricted to intra-node NVLink domains ($TP \le 8$).

---

## 4. Control Plane & Profiler Scalability

To accurately profile distributed clusters up to 1,000 QPS without client-side bottlenecks:

### 4.1. Traffic Generator Load
- Target load: 500 concurrent requests, generating 100 output tokens each.
- Total streamed gRPC response chunks: $500 \times 100 = 50,000$ messages over a 5-second generation window.
- Inbound packet rate: $\approx 10,000 \text{ msgs/sec}$.
- Go gRPC Client Resource Consumption:
  - 500 Goroutines with buffered channels require ~2.5 MB RAM (at ~4 KB / goroutine).
  - Network I/O: 10,000 chunks/sec $\times$ 128 bytes/chunk $\approx 1.28 \text{ MB/s}$ (negligible on 10GbE network).
  - CPU utilization: ~1.5 to 2 cores of modern x86-64 CPU for parsing protobuf frames and recording nanosecond timestamps.

### 4.2. Telemetry Ingestion (Prometheus & DCGM)
- DCGM sample interval: 100 ms (10 Hz) for high-resolution burst detection.
- Metrics per GPU: 12 signals (utilization, fb_used, sm_clock, power, pcie_rx/tx, nvlink_rx/tx, throttle_reasons).
- Cluster size: 8 GPUs $\implies 8 \times 12 \times 10 = 960$ metric data points per second.
- Prometheus scrape interval: 1s with local in-memory buffer $\implies \approx 1 \text{ MB}$ raw timeseries per 10-minute experiment run.

---

## 5. Summary Sizing Matrix for Profiling Experiments

| Experiment Configuration | Target Hardware | Max Safe Batch Size ($S=4096$) | Expected TTFT ($p50$) | Expected ITL ($p50$) | Target Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama-3-8B FP16 (TP=1)** | 1x A100-80GB | 96 | 85 ms | 11.2 ms | ~1,200 tok/s |
| **Llama-3-8B FP16 (TP=2)** | 2x A100-80GB (NVLink) | 192 | 52 ms | 6.8 ms | ~2,300 tok/s |
| **Llama-3-8B FP8 (TP=1)** | 1x H100-80GB | 192 | 38 ms | 4.9 ms | ~3,400 tok/s |
| **Llama-3-70B FP8 (TP=4)**| 4x H100-80GB (NVLink) | 128 | 92 ms | 8.1 ms | ~4,800 tok/s |
| **Llama-3-70B FP16 (TP=8)**| 8x H100-80GB (NVLink) | 256 | 65 ms | 5.4 ms | ~8,200 tok/s |
