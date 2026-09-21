# CUDA atomic add benchmark

A native CUDA C++ counterpart to the `wgsl_rust` random-access `atomicAdd` benchmark. It preserves
the same hash, seed, 256-thread groups, buffer-size sweep, warmup, per-thread checksum writes,
sample count, and full counter-sum validation.

## Build and run

Install CMake 3.24 or newer and the CUDA toolkit, then:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cuda-atomic-add-bench
```

CMake targets the build machine's native GPU architecture by default. Override it when building
for another GPU, for example with `-DCMAKE_CUDA_ARCHITECTURES=89` for Ada (`sm_89`). A direct build
is also possible:

```sh
nvcc -O3 -std=c++17 -arch=native main.cu -o cuda-atomic-add-bench
./cuda-atomic-add-bench
```

The benchmark sweeps powers of two from 128 bytes through `2^27` bytes (128 MiB). For every power
of two it also measures buffers approximately 28% smaller and 28% larger, rounded to whole `u32`
elements. Each of the 63 cases runs five measured samples and is fully validated. Options:

```text
--device N       zero-based CUDA device index (default: 0)
--workgroups N   launched CUDA blocks (default: 4096)
--adds N         atomic adds per thread (default: 1024)
--samples N      measured samples per buffer size (default: 5)
```

Progress, device information, and validation status are written to stderr. Once the sweep finishes,
stdout contains CSV, so results can be captured without progress messages:

```sh
./build/cuda-atomic-add-bench > results.csv
```

The original `average_gatomic_per_s` and `median_gatomic_per_s` columns use host wall-clock time
from immediately before kernel launch through completion, making them the closest comparison with
the WGSL benchmark. The additional `average_gpu_gatomic_per_s` and `median_gpu_gatomic_per_s`
columns use CUDA events to isolate GPU execution time. Buffer clearing, warmup, host readback, and
validation are outside both timed regions.

The result includes integer hashing and modulo address generation performed by the kernel. It is a
random-access CUDA workload measurement, rather than an isolated hardware atomic instruction rate.
