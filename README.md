# GPU random atomic-add microbenchmark

This repository contains the same random-access `atomicAdd` workload in WGSL/wgpu, CUDA C++,
and Vulkan C++.
It is intended as a small reproducer for comparing the workload across operating-system and API
combinations. It is not a general benchmark suite.

Each of 4096 workgroups has 256 threads. Every thread performs 1024 atomic increments at hashed
locations in a `u32` buffer. The benchmark sweeps buffer sizes from 256 KiB to 64 MiB in ten
logarithmic steps per octave. A warmup runs first, buffer clearing is not timed, and the final sum
of all counters is checked after every buffer size.

## Run

WGSL through Vulkan:

```sh
cargo run --release --manifest-path wgsl_rust/Cargo.toml > wgsl-vulkan.csv
```

WGSL through Direct3D 12 (Windows only):

```sh
cargo run --release --manifest-path wgsl_rust/Cargo.toml -- --backend dx12 > wgsl-dx12.csv
```

CUDA:

```sh
cmake -S cuda_cpp -B cuda_cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cuda_cpp/build -j
./cuda_cpp/build/cuda-atomic-add-bench > cuda.csv
```

Vulkan C++ (requires Vulkan headers and loader plus `glslc` or `glslangValidator`):

```sh
cmake -S vulkan_cpp -B vulkan_cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build vulkan_cpp/build -j
./vulkan_cpp/build/vulkan-atomic-add-bench > vulkan-cpp.csv
```

On Windows, the default Visual Studio generator is multi-configuration, so select Release while
building and run the executable from its configuration directory:

```powershell
cmake -S cuda_cpp -B cuda_cpp/build
cmake --build cuda_cpp/build --config Release -j
.\cuda_cpp\build\Release\cuda-atomic-add-bench.exe > cuda.csv
```

The same applies to the Vulkan C++ target:

```powershell
cmake -S vulkan_cpp -B vulkan_cpp/build
cmake --build vulkan_cpp/build --config Release -j
.\vulkan_cpp\build\Release\vulkan-atomic-add-bench.exe > vulkan-cpp.csv
```

The installed NVIDIA driver must support the CUDA toolkit used to build the benchmark. If the
versions reported by `nvidia-smi` and `nvcc --version` are incompatible, update the driver or select
an older installed toolkit with `-DCMAKE_CUDA_COMPILER=path/to/nvcc`.

The CUDA CMake target uses the current machine's native GPU architecture by default. Set
`-DCMAKE_CUDA_ARCHITECTURES=NN` when building for a different GPU.

The Vulkan executable uses physical device 0 by default. Pass `--device N` to select a different
device from Vulkan's enumeration order. It compiles the GLSL compute shader to SPIR-V at build time.

Progress and GPU information go to stderr. Standard output contains a short metadata preamble and
then two-column CSV data. This keeps each result self-describing without repeating the environment,
API, device, and driver on every row. Close other GPU-heavy programs and use a stable power setting
when collecting results.

All versions use a CPU wall clock from command submission through GPU completion. Pipeline setup,
warmup, buffer clearing, readback, and validation are outside the timed region.

To graph every CSV in this directory (requires matplotlib):

```sh
python3 plot_throughput.py
```

The graph legend is built from each result's metadata. Pass explicit filenames to plot only a
subset, or use `-o FILE` to choose the output path.

The workload includes hashing and modulo address calculation. Reported throughput is therefore the
rate of this complete random-access workload, not the isolated hardware atomic instruction rate.

## Optional arguments

All executables accept `--workgroups N`, `--adds N`, and `--samples N`. The WGSL version also
accepts `--backend vulkan|dx12`; the CUDA and Vulkan C++ versions accept `--device N`. Defaults are
encoded near the top of each source file.
