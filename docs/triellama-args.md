# TrieLlama argument reference

This page documents CLI arguments and environment variables **introduced and modified directly by TrieLlama**:
- **Hierarchical Radix Prefix Caching (RXC)** arguments
- **Streaming Norm-Calibrated (SNC) KV Cache** formats
- **Target Platform Build Flags**

---

> [!TIP]
> ### Looking for Inherited BeeLlama or Upstream llama.cpp Arguments?
> For arguments inherited from BeeLlama, please consult the upstream BeeLlama documentation:
> - **[BeeLlama Argument Reference](https://github.com/Anbeeld/beellama.cpp/blob/main/docs/beellama-args.md)** — Comprehensive reference for:
>   - `--cache-type-k kvarn*` / `--cache-type-v kvarn*` (KVarN 2-8 bit KV-cache)
>   - `--kv-tail-tokens N` (Precision tail tokens)
>   - `--spec-dm-controller profit` (Adaptive DFlash draft-max controller)
>   - `-ubd N, --spec-draft-ubatch-size N` (Independent draft micro-batch size)
>   - `--reasoning-loop-guard` (Reasoning loop protection mode)
>   - `--models-preset` (INI model presets)
>   - `--spec-synth-len` and KLD perplexity measurement flags
>
> For general upstream `llama.cpp` arguments, run `llama-server --help` or `llama-cli --help`.

---

## 1. Hierarchical Radix Caching (RXC) Arguments

TrieLlama provides fine-grained control over multi-tier memory management across GPU VRAM, Host System RAM, and NVMe SSD:

| Argument | Environment variable | Default | Description |
|---|---|---|---|
| `--cache-disk-dir PATH` | `LLAMA_ARG_CACHE_DISK_DIR` | `${XDG_CACHE_HOME:-~/.cache}/triellama.cpp/radix` | Absolute directory path where persistent KV checkpoints and trie metadata are stored. Subdirectories are isolated automatically per model architecture, quantization, and tensor shape fingerprint. |
| `--cache-disk-max-bytes BYTES` | `LLAMA_ARG_CACHE_DISK_MAX_BYTES` | `0` (unlimited) | Global disk storage ceiling. Supports human-readable size suffixes: `B`, `K`/`KB`/`KiB`, `M`/`MB`/`MiB`, `G`/`GB`/`GiB`, `T`/`TB`/`TiB` (e.g. `64GB`). When total checkpoint size exceeds this threshold, the LRU scavenger prunes unpinned oldest leaf checkpoints. |
| `-cram N`, `--cache-ram N` | `LLAMA_ARG_CACHE_RAM` | `40960` | Maximum token capacity kept staged in Host System RAM. Set to `0` for **Zero-RAM Offload Mode**, which skips RAM buffering entirely and streams inactive KV checkpoints directly to/from NVMe SSD. |
| `--checkpoint-min-step N` | `LLAMA_ARG_CHECKPOINT_MIN_STEP` | `128` | Minimum token delta between turn checkpoints. Prevents excessive I/O thrashing on rapid, short generation bursts. |

### Example Server Configuration

```bash
# Launch server with 64GB persistent NVMe cache and Zero-RAM offload
llama-server \
  -m models/Qwen3.6-27B.gguf \
  --port 8080 -c 32768 -np 4 \
  --cache-disk-dir ~/.cache/triellama.cpp/radix \
  --cache-disk-max-bytes 64GB \
  -cram 0
```

---

## 2. Streaming Norm-Calibrated (SNC) Cache Formats

TrieLlama introduces `snc4` and `snc8` as selectable KV-cache formats alongside standard upstream types and KVarN:

| Argument | Values | Description |
|---|---|---|
| `-ctk TYPE`, `--cache-type-k TYPE` | `snc4`, `snc8` | Selects Streaming Norm-Calibrated quantization for target K cache. `snc4` uses 4-bit quants with fp16 calibrated scales (4.500 bpe); `snc8` uses 8-bit quants with fp16 calibrated scales (8.500 bpe). |
| `-ctv TYPE`, `--cache-type-v TYPE` | `snc4`, `snc8` | Selects Streaming Norm-Calibrated quantization for target V cache. |
| `-ctkd TYPE`, `--spec-draft-type-k TYPE` | `snc4`, `snc8` | Selects SNC quantization for owned model-backed speculative draft K cache. |
| `-ctvd TYPE`, `--spec-draft-type-v TYPE` | `snc4`, `snc8` | Selects SNC quantization for owned model-backed speculative draft V cache. |

> [!NOTE]
> Unlike KVarN (which uses structured 128-token tiles and requires rectangular head dimensions for D64), SNC is a conventional row-blocked quantizer (QK=32). It can be used across arbitrary context sizes and head dimensions.

---

## 3. CMake Build Configuration Flags

TrieLlama concentrates hardware acceleration on **CUDA**, **ROCm/HIP**, and **CPU**:

| CMake Flag | Default | Description |
|---|---|---|
| `-DGGML_CUDA=ON` | `OFF` | Enables NVIDIA CUDA backend with FlashAttention kernels covering standard, KVarN, and SNC pairs. |
| `-DGGML_HIP=ON` | `OFF` | Enables AMD ROCm / HIP backend with FlashAttention and hipBLAS acceleration for Radeon and Strix Halo APUs. |
| `-DGGML_NATIVE=ON` | `ON` | Optimizes host binary for native CPU instruction set (AVX2, AVX-512, NEON). |
| `-DGGML_CUDA_FA=ON` | `ON` | Enables FlashAttention vector and prefill kernels on CUDA. |
| `-DGGML_CUDA_FA_ALL_QUANTS=ON` | `OFF` | Expands CUDA FlashAttention to compile all 169 standard and SNC pair combinations. |
