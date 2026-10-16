# TrieLlama Features

**TrieLlama.cpp** is built upon [Anbeeld's BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) and [Georgi Gerganov's llama.cpp](https://github.com/ggml-org/llama.cpp).

This document covers features **introduced and modified directly by TrieLlama**:
- **Hierarchical Radix Prefix Caching (RXC)** across GPU VRAM, System RAM, and NVMe SSD
- **Experimental Streaming Norm-Calibrated (SNC) KV Cache Quantization** (`snc4` / `snc8`)
- **Focused Hardware Acceleration** targeting CUDA, ROCm/HIP, and CPU

---

> [!TIP]
> ### Looking for Inherited BeeLlama Features?
> For documentation, technical write-ups, and benchmarks for features inherited from BeeLlama, please consult the upstream BeeLlama documentation:
> - **[BeeLlama Features Reference](https://github.com/Anbeeld/beellama.cpp/blob/main/docs/beellama-features.md)** — KVarN quantization (`kvarn2`..`kvarn8`), precision tails (`--kv-tail-tokens`), adaptive DFlash speculative decoding (`--spec-dm-controller profit`), independent draft batching (`-ubd`), reasoning-loop protection, and KLD perplexity measurement.
> - **[BeeLlama Argument Reference](https://github.com/Anbeeld/beellama.cpp/blob/main/docs/beellama-args.md)** — Comprehensive CLI reference for BeeLlama options.
> - **[BeeLlama INI Presets](https://github.com/Anbeeld/beellama.cpp/blob/main/docs/preset.md)** — Multi-model preset configuration format.
> - **[Anbeeld's Research Articles](https://anbeeld.com/articles)** — In-depth mathematical formulations and Wikitext benchmarks for KVarN and precision tails.

---

## 1. Hierarchical Radix Prefix Caching (RXC)

### What it is
In upstream `llama.cpp` and BeeLlama, prompt cache reuse is managed through flat linear lists. Finding a reusable prefix requires an $O(N \cdot L)$ scan across all cached sequences, with no ability to detect or share branching sub-prefixes across concurrent requests or multi-turn chats.

TrieLlama introduces **Hierarchical Radix Prefix Caching (RXC)**, replacing flat lists with a token-level **Radix Tree** (compressed prefix trie):

```
                   [ Root Node ]
                         |
                 "System Prompt..." (Edge: 256 tokens)
                         |
                  [ Branch Node 1 ]
                 /                 \
     "User 1: ..."                  "User 2: ..."
     (Edge: 128 tok)                (Edge: 128 tok)
           |                              |
    [ Checkpoint A ]               [ Checkpoint B ]
    (Turn 1 KV State)              (Turn 1 KV State)
           |                              |
    "Assistant 1: ..."             "Assistant 2: ..."
```

Prefix lookup executes in $O(L)$ time proportional only to the prompt length, completely independent of how many thousands of conversation turns are cached.

### 3-Tier Storage Hierarchy
RXC decouples KV-cache retention from physical GPU VRAM by managing three tiered memory layers:

1. **Tier 1: GPU VRAM (Active Contexts):**
   - Zero-latency attention compute directly on device.
   - Actively managed across model context slots.
2. **Tier 2: Host System RAM (Staged Prefixes):**
   - High-speed DRAM buffering for warm conversation branches.
   - Governed by `--cache-ram` (`-cram`) token limit.
3. **Tier 3: Persistent NVMe SSD Disk (Cold Checkpoints):**
   - Multi-gigabyte persistent cache stored under `--cache-disk-dir`.
   - Survives server restarts and hydrates automatically on cold starts.
   - Governed by `--cache-disk-max-bytes` with background LRU leaf pruning.

### Key Capabilities & Invariants
- **Multi-Branch Prefix Sharing:** Independent slots and requests sharing common system prompts, multi-tenant templates, or shared documents reuse existing KV states without prefill overhead.
- **Cross-Restart Disk Persistence:** Valid prefix nodes are indexed and written to disk checkpoints. Upon server restart, existing checkpoints hydrate immediately into the trie, eliminating cold-start prefill delay.
- **Zero-RAM Offload Mode (`-cram 0`):** In memory-constrained environments or dedicated GPU nodes, `-cram 0` streams inactive KV states directly between GPU VRAM and NVMe SSD, keeping host RAM consumption strictly minimal.
- **Global Disk Quota & LRU Scavenging (`--cache-disk-max-bytes`):** Enforces a strict upper bound on disk usage. When storage exceeds the limit, the scavenger prunes the oldest unpinned leaf branches to prevent disk bloat.
- **KV-Tail & KVarN Invariant Preservation:** Fully compatible with KVarN 128-token chunk boundaries (`KVAR_N_GROUP = 128`) and preserves homogeneous F16/BF16 precision tails across transactional restores.
- **Multimodal Prefix Caching:** Indexes and caches multimodal vision and audio embedding prefixes alongside text tokens.
- **SWA Degraded Recovery:** Gracefully truncates and restores sliding-window attention (SWA) caches without corrupting history.

For the full architectural walkthrough and benchmarks, see [Radix Cache Walkthrough](radix-cache-walkthrough.md).

---

## 2. Streaming Norm-Calibrated (SNC) KV Caches

> [!NOTE]
> **Experimental Status**: SNC is an experimental, vibe-engineered heuristic created for personal hardware rig experimentation rather than a peer-reviewed academic publication. While it demonstrates solid norm-preservation characteristics in local testing, users requiring academically validated quality benchmarks should prefer KVarN or standard `q8_0`/`q4_0`.

### What it is
Streaming Norm-Calibrated (SNC) quantization introduces `snc4` (4.500 bpe, 4-bit, QK=32) and `snc8` (8.500 bpe, 8-bit, QK=32) for high-fidelity KV cache compression without requiring full Hadamard rotations or offline calibration datasets.

Standard uniform quantizers (`q4_0`, `q8_0`) suffer from systematic norm contraction under inner products. SNC dynamically calibrates block scales against the total row L2 norm:

$$\gamma = \frac{\|x\|_2}{\|\hat{x}_0\|_2} = \sqrt{\frac{\sum x_i^2}{\sum \hat{x}_{0,i}^2}}$$

$$d^* = d_0 \cdot \gamma$$

The calibration ratio $\gamma$ is folded directly into the stored 16-bit half scale $d^*$, preserving exact block L2 norms and achieving near-zero dot product expectation error ($\mathbb{E}_M \approx 0$) without any auxiliary scale or norm metadata overhead:
- `snc4`: 18 bytes per 32 elements (16 bytes 4-bit packed nibbles + 2 bytes fp16 calibrated scale).
- `snc8`: 34 bytes per 32 elements (32 bytes 8-bit signed quants + 2 bytes fp16 calibrated scale).

### When to use them
- Use SNC when experimenting with long-context inference (e.g. 64k, 128k, 256k) where standard `q4_0` or `q8_0` accumulate dot-product drift, or when KVarN's structured 128-token tile representation cannot be used.
- Recommended pairing: `snc4 K + snc8 V` for high fidelity, or `snc4 K + snc4 V` for balanced 50% memory reduction relative to 8-bit caches.

### Backend Support
- **CPU**: SIMD vector kernels for AVX2, AVX-512, and ARM NEON for streaming quantize and dequantize.
- **CUDA / ROCm (HIP)**: Native vector FlashAttention (`fattn-vec`) covering all homogeneous and mixed pairs (`snc4:snc4`, `snc8:snc8`, `snc4:snc8`, `snc8:snc4`, `snc4:f16`, `snc8:f16`, `f16:snc4`, `f16:snc8`, `snc4:bf16`, `snc8:bf16`, `bf16:snc4`, `bf16:snc8`), prefill tile FlashAttention, `SET_ROWS`, and `GET_ROWS`.
- **Radix Caching & SWA**: Fully compatible with hierarchical multi-tier Radix caching (RXC) and turn-based disk checkpoints.

---

## 3. Platform Focus & Build Surface

TrieLlama concentrates performance engineering strictly on **CUDA**, **ROCm/HIP** (AMD Radeon and Strix Halo APUs), and **CPU**:
- Pruned unmaintained and niche backends (MUSA, SYCL, CANN, OpenVINO, Snapdragon, etc.).
- Removed prebuilt release automation and container wrappers in favor of direct declarative and CMake source builds.
