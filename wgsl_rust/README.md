# WGSL atomic add benchmark

A deliberately small native GPU benchmark for random-access `atomicAdd` operations over `u32`
storage buffers. Each operation hashes a unique invocation/iteration key to select a uniformly
distributed slot. The returned old values feed per-thread checksums, preventing the operations from
being collapsed.

## Run

Install the current stable Rust toolchain, then:

```sh
cargo run --release
```

Vulkan is used by default on every platform. Select Direct3D 12 with `--backend dx12` on Windows.
The first build downloads and compiles `wgpu`.
The benchmark sweeps powers of two from 128 bytes through `2^27` bytes (128 MiB). For every power of
two it also measures buffers approximately 28% smaller and 28% larger, rounded to whole `u32`
elements. Each of the 63 cases runs five measured samples and is fully validated. Options:

```text
--backend NAME   vulkan or dx12 (default: vulkan)
--workgroups N   dispatched workgroups (default: 4096)
--adds N         atomic adds per thread (default: 1024)
--samples N      measured samples per buffer size (default: 5)
```

Progress and validation status are written to stderr. Once the sweep finishes, stdout contains CSV,
with average and median throughput for each buffer size. Results can be captured without the
progress messages:

```sh
cargo run --release > results.csv
```

Close GPU-heavy applications and use a stable GPU power/performance setting for repeatable results.

The timer covers command submission through GPU completion. Command encoding, pipeline creation,
shader compilation, warmup, buffer clearing, result copying, mapping, and validation are outside the
measured interval. Validation sums the entire atomic buffer after every case.

The result includes integer hashing and modulo address generation performed by the shader. It is a
random-access WGSL workload measurement, rather than an isolated hardware atomic instruction rate.
