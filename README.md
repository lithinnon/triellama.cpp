# TrieLlama.cpp

**TrieLlama.cpp** (or just **TrieLlama**) brings native **Hierarchical Radix Prefix Caching (RXC)** across GPU VRAM, System RAM, and NVMe SSD to `llama.cpp`.

By replacing flat linear prompt caches with a high-performance token-level prefix trie, TrieLlama enables instant prompt prefix sharing across slots, zero-RAM NVMe disk offloading, cross-restart cache persistence, and strict LRU disk quota management.

> **Acknowledgments & Lineage:** TrieLlama is built upon [Anbeeld's BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) and [Georgi Gerganov's llama.cpp](https://github.com/ggml-org/llama.cpp). We gratefully acknowledge Anbeeld's foundational contributions in KVarN quantization, precision tails, and speculative decoding. Please consider supporting Anbeeld's independent research:

[![Support Anbeeld's work!](https://anbeeld.com/images/support.jpg)](https://anbeeld.com/support)

---

## Flagship Feature: Hierarchical Radix Prefix Caching (RXC)

Standard `llama.cpp` stores cached prompt states in a flat list, requiring an $O(N \cdot L)$ scan across all cached sequences to find the longest common prefix. Under concurrent multi-turn chat sessions or repeated system prompts, this leads to heavy redundant KV-cache compute, search latency, and high VRAM pressure.

**TrieLlama's Radix Prefix Cache (RXC)** structures cached states into a token-level **Radix Tree** (compressed prefix trie):

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

Prefix lookup runs in $O(L)$ time (proportional only to the prompt length), completely decoupled from the number of active conversation branches.

### 3-Tier Storage Hierarchy

RXC decouples KV-cache capacity from physical GPU memory by managing a 3-tier hierarchy:

```
+-------------------------------------------------------------------------+
| Tier 1: GPU VRAM (Active Contexts)                                      |
|  - Zero-latency attention compute directly on device                    |
|  - Managed via model context slots                                      |
+------------------------------------+------------------------------------+
                                     | (Evict / Restore on Turn Completion)
                                     v
+-------------------------------------------------------------------------+
| Tier 2: Host System RAM (Staged Prefixes)                               |
|  - High-speed DRAM buffering for warm conversation branches             |
|  - Governed by `--cache-ram` (`-cram`) token limit                      |
+------------------------------------+------------------------------------+
                                     | (Demote / Hydrate on Demand)
                                     v
+-------------------------------------------------------------------------+
| Tier 3: Persistent NVMe / SSD Disk (Cold Checkpoints)                   |
|  - Multi-gigabyte persistent cache surviving server restarts            |
|  - Managed via `--cache-disk-dir` with LRU quota scavenging             |
+-------------------------------------------------------------------------+
```

### Key Capabilities

- **Instant Multi-Branch Prefix Sharing:** Independent slots and requests sharing common system prompts, document chunks, or conversation histories reuse existing KV states immediately without prefill delay.
- **Cross-Restart Disk Persistence:** Valid prefix nodes are indexed and persisted under `--cache-disk-dir`. When the server is restarted, cold checkpoints are immediately hydrated into the trie, eliminating cold-start prefill on subsequent sessions.
- **Zero-RAM Offload Mode (`-cram 0`):** In memory-constrained environments or dedicated GPU nodes, set `-cram 0` to stream inactive KV states directly between GPU VRAM and NVMe SSD, keeping host RAM consumption strictly minimal.
- **Global Disk Quota & LRU Scavenging (`--cache-disk-max-bytes`):** Enforces a strict upper bound on disk usage. When storage exceeds the limit, the background scavenger prunes the oldest unpinned leaf branches to prevent unbounded disk usage.
- **KV-Tail & KVarN Invariant Preservation:** Fully compatible with KVarN 128-token chunk boundaries (`KVAR_N_GROUP = 128`) and preserves homogeneous F16/BF16 precision tails across transactional restores.
- **Multimodal Prefix Caching:** Indexes and caches multimodal vision and audio embedding prefixes alongside text tokens.

For an in-depth architectural guide and benchmark details, see the [Radix Cache Walkthrough](docs/radix-cache-walkthrough.md).

---

## Quick Start: Radix Caching

Launch `llama-server` with hierarchical radix caching enabled:

```bash
# Launch server with persistent NVMe caching and zero-RAM mode
llama-server \
  -m /path/to/model.gguf \
  --port 8080 \
  -c 32768 -np 4 \
  --cache-disk-dir ~/.cache/triellama.cpp/radix \
  --cache-disk-max-bytes 64GB \
  -cram 0
```

### Key Radix Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--cache-disk-dir PATH` | `${XDG_CACHE_HOME:-~/.cache}/triellama.cpp/radix` | Directory where persistent KV checkpoints and trie metadata are stored. |
| `--cache-disk-max-bytes BYTES` | `0` (unlimited) | Global disk storage limit (supports suffixes like `32GB`, `500MB`). Triggers LRU eviction when exceeded. |
| `-cram, --cache-ram N` | `40960` | Maximum tokens kept in Host System RAM. Set to `0` for zero-RAM direct-to-disk offload. |
| `--checkpoint-min-step N` | `128` | Minimum token interval between turn checkpoints. |

---

## Inherited & Experimental Features

TrieLlama inherits the core performance features introduced by Anbeeld in BeeLlama, along with custom experimental additions:

- **Variance-Normalized KV-Cache Quantization (KVarN):** Calibration-free 2- to 8-bit KV-cache quantization (`kvarn2`..`kvarn8`) for high-fidelity compression.
- **KV Cache Precision Tail:** Keeps historical tokens quantized while retaining the newest tokens in exact F16/BF16 (`--kv-tail-tokens`).
- **Standard Low-Bit KV Formats:** Extends upstream cache types with `q2_0`, `q2_1`, `q3_0`, `q3_1`, `q6_0`, and `q6_1`.
- **Adaptive DFlash Speculative Decoding:** Dynamic speculative draft-horizon controller (`--spec-dm-controller profit`).
- **Reasoning-Loop Protection:** Prevents infinite generation loops in reasoning models (`--reasoning-loop-guard`).
- **Experimental SNC Quantization (`snc4` / `snc8`):** A lightweight, norm-calibrated KV cache quantization format crafted for local hardware experimentation. It is strictly an experimental, vibe-engineered heuristic to satisfy personal rig constraints—use KVarN or standard `q8_0`/`q4_0` if you need benchmark-verified fidelity.

> For full technical write-ups, mathematical derivations, Wikitext benchmark ladders, and preset formats for inherited features, please refer to the original **[BeeLlama.cpp repository](https://github.com/Anbeeld/beellama.cpp)** and our local [TrieLlama Features Matrix](docs/triellama-features.md).

---

## Build from Source

TrieLlama focuses development, optimization, and testing strictly on **CUDA**, **ROCm/HIP**, and **CPU** backends:

### CUDA Build (Linux / Windows)

```bash
# Linux (GCC + CUDA)
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Windows (MSVC + CUDA)
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON ^
  -DGGML_CUDA_FA=ON ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

### ROCm / HIP Build (AMD Radeon & Strix Halo APUs)

```bash
cmake -B build -DGGML_HIP=ON -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### CPU Build

```bash
cmake -B build -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

---

## Documentation

- [Hierarchical Radix Cache Walkthrough](docs/radix-cache-walkthrough.md) — Architectural design, eviction policies, and benchmarks
- [TrieLlama Features Matrix](docs/triellama-features.md) — Detailed feature list and upstream diff
- [Argument Reference](docs/triellama-args.md) — Comprehensive CLI flag reference
- [Speculative Decoding](docs/speculative.md) — Speculative decoding options and controllers
- [HTTP Server Documentation](tools/server/README.md) — REST API endpoints, slots, and web UI
- [Performance Troubleshooting](docs/development/token_generation_performance_tips.md) — Optimization guide

## Contributing

Keep PRs small and scoped. Run relevant tests before opening a PR against `lithinnon/triellama.cpp`:

```bash
./build/bin/test-radix-tree
./build/bin/test-arg-parser
python3 tests/test-upstream-merge-keepers-static.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) for contribution guidelines, AI usage policies, and private security disclosures.

## License & Credits

TrieLlama is licensed under the MIT License. See [LICENSE](LICENSE) for details.

- Built upon [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and the `ggml-org` team.
- Built upon [BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) by Anbeeld.
