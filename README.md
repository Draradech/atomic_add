# GPU random atomic-add microbenchmark

This repository contains the same random-access `atomicAdd` workload in WGSL/wgpu, CUDA C++,
and Vulkan C++.
It is intended as a small reproducer for comparing the workload across operating-system and API
combinations. It is not a general benchmark suite.

Each of 4096 workgroups has 256 threads. Every thread performs 1024 atomic increments at hashed
locations in a `u32` buffer. The sweep variants test buffer sizes from 256 KiB to 64 MiB in ten
logarithmic steps per octave. A warmup runs first, buffer clearing is not timed, and the final sum
of all counters is checked after every buffer size.

## Run

WGSL through Vulkan:

```sh
cargo run --release --manifest-path wgsl_rust/Cargo.toml > wgsl-vulkan.csv
```

For a dedicated Vulkan allocation while keeping the WGSL shader and wgpu
compute pipeline, add `--dedicated-counter`:

```sh
cargo run --release --manifest-path wgsl_rust/Cargo.toml -- --dedicated-counter > wgsl-vulkan-dedicated.csv
```

This opt-in path uses wgpu's unsafe Vulkan HAL interop to create, initialize,
and import the counter buffer. It requires a Vulkan wgpu device. It is a
platform-specific escape hatch for the Linux NVIDIA behavior described below.

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

Pass `--small-buffers` to the Vulkan C++ sweep to create a counter buffer and
backing allocation for each sweep size. The default reuses one 64 MiB counter
buffer and allocation for the whole sweep. Both modes clear and validate every
size; CSV output from the small-buffer mode includes
`# counter_allocation=per-step` so plots can distinguish the runs.

```sh
./vulkan_cpp/build/vulkan-atomic-add-bench --small-buffers > vulkan-cpp-small.csv
```

On the tested Linux RTX 4070/NVIDIA 615.71.09 driver, `--dedicated-counter`
recovers the fast random-atomic path. It attaches
`VkMemoryDedicatedAllocateInfo` for the counter buffer when allocating its
`VkDeviceMemory`. It works with either the default 64 MiB buffer or
`--small-buffers`; the latter gives each counter buffer a dedicated allocation
sized to its Vulkan memory requirements. The CSV includes
`# counter_memory=dedicated`.

```sh
./vulkan_cpp/build/vulkan-atomic-add-bench --dedicated-counter > vulkan-cpp-dedicated.csv
./vulkan_cpp/build/vulkan-atomic-add-bench --small-buffers --dedicated-counter > vulkan-cpp-small-dedicated.csv
```

At 8 MiB, the checksum-validating Linux sweep measured 29.4 Gatomic/s with a
dedicated 64 MiB counter allocation and 25.0 Gatomic/s with an 8 MiB per-step
dedicated allocation (three samples each), compared with 12.5 Gatomic/s in the
earlier non-dedicated sweeps (seven samples each). The driver-level reason for
the difference is not yet known; see `investigation.txt`.

The WGSL/wgpu sweep also benefits from the dedicated counter buffer: at
8 MiB it measured 31.3 Gatomic/s in a three-sample run and 22.1 Gatomic/s
in a seven-sample run, versus 12.3 Gatomic/s without dedicated memory in a
three-sample run. Its dedicated results varied more than the Vulkan C++ runs;
both runs passed validation at every size.

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

## Continuous 8 MiB Vulkan run

The separate `vulkan_cpp_single` variant runs its own shader without the checksum path on an
8 MiB counter buffer continuously. It reports completed atomic adds divided by elapsed wall time
about every five seconds. It does not clear or read back buffers, validate results, or write CSV.
Stop it with Ctrl-C.
The 8 MiB buffer uses a 64 MiB device-memory allocation; see `investigation.txt` for the
performance measurements behind that choice.
Change the device using the constant in `vulkan_cpp_single/main.cpp`. Keep the workload constants
in that file and `vulkan_cpp_single/atomic_add.comp` aligned when changing workgroups, adds per
thread, or buffer size.

```sh
cmake -S vulkan_cpp_single -B vulkan_cpp_single/build -DCMAKE_BUILD_TYPE=Release
cmake --build vulkan_cpp_single/build -j
./vulkan_cpp_single/build/vulkan-atomic-add-single
```

On Windows, build with `--config Release` and run
`vulkan_cpp_single\build\Release\vulkan-atomic-add-single.exe`.

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

All sweep versions use a CPU wall clock from command submission through GPU completion. Pipeline setup,
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

The three sweep executables accept `--workgroups N`, `--adds N`, and `--samples N`. The WGSL version also
accepts `--backend vulkan|dx12` and `--dedicated-counter`; the CUDA and Vulkan C++ versions accept `--device N`.
Defaults are encoded near the top of each source file. The Vulkan C++ version also accepts
`--small-buffers` and `--dedicated-counter`.
