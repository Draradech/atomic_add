#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kBlockSize = 256;
constexpr std::uint32_t kMinBufferExponent = 18;  // 256 KiB
constexpr std::uint32_t kMaxBufferExponent = 26;  // 64 MiB
constexpr std::uint32_t kStepsPerOctave = 10;
constexpr std::uint32_t kSeed = 0x12345678u;

constexpr std::array<const char*, kStepsPerOctave> kStepRelations = {
    "power_of_two", "step_1_of_10", "step_2_of_10", "step_3_of_10", "step_4_of_10",
    "step_5_of_10", "step_6_of_10", "step_7_of_10", "step_8_of_10", "step_9_of_10"};

void cuda_check(cudaError_t result, const char* expression, const char* file, int line) {
    if (result != cudaSuccess) {
        const char* error_name = cudaGetErrorName(result);
        const char* error_message = cudaGetErrorString(result);
        if (result == cudaErrorInsufficientDriver) {
            error_name = "cudaErrorInsufficientDriver";
            error_message =
                "the installed NVIDIA driver is too old for this CUDA runtime; update the "
                "driver or rebuild with an older CUDA toolkit";
        }
        throw std::runtime_error(
            std::string(expression) + " failed at " + file + ":" + std::to_string(line) +
            ": " + (error_name != nullptr ? error_name : "unknown CUDA error") + " (" +
            std::to_string(static_cast<int>(result)) + "): " +
            (error_message != nullptr ? error_message : "no error description available"));
    }
}

#define CUDA_CHECK(expression) cuda_check((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }
    std::size_t count() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class Event {
public:
    Event() { CUDA_CHECK(cudaEventCreate(&event_)); }

    ~Event() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

struct Config {
    std::uint32_t workgroups = 4096;
    std::uint32_t adds_per_thread = 1024;
    std::uint32_t samples = 5;
    std::uint32_t device = 0;
};

struct Case {
    std::uint64_t buffer_size;
    const char* relation;
    std::uint32_t exponent;
};

struct Result {
    Case benchmark_case;
    double host_average;
    double host_median;
    double gpu_average;
    double gpu_median;
};

void usage(const char* program) {
    std::cerr << "Usage: " << program
              << " [--device N] [--workgroups N] [--adds N] [--samples N]\n"
                 "Defaults: --device 0 --workgroups 4096 --adds 1024 --samples 5\n";
}

std::uint32_t parse_u32(const std::string& flag, const char* value, bool allow_zero) {
    if (value[0] == '\0' || value[0] == '-') {
        throw std::runtime_error("invalid integer for " + flag + ": " + value);
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' ||
        parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid integer for " + flag + ": " + value);
    }
    if (parsed == 0 && !allow_zero) {
        throw std::runtime_error(flag + " must be greater than zero");
    }
    return static_cast<std::uint32_t>(parsed);
}

Config parse_args(int argc, char** argv) {
    Config config;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        if (flag == "-h" || flag == "--help") {
            usage(argv[0]);
            std::exit(0);
        }
        if (index + 1 == argc) {
            throw std::runtime_error("missing value after " + flag);
        }
        if (flag == "--device") {
            config.device = parse_u32(flag, argv[++index], true);
        } else if (flag == "--workgroups") {
            config.workgroups = parse_u32(flag, argv[++index], false);
        } else if (flag == "--adds") {
            config.adds_per_thread = parse_u32(flag, argv[++index], false);
        } else if (flag == "--samples") {
            config.samples = parse_u32(flag, argv[++index], false);
        } else {
            throw std::runtime_error("unknown option: " + flag);
        }
    }
    return config;
}

__device__ __forceinline__ std::uint32_t hash_u32(std::uint32_t input) {
    std::uint32_t value = input;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

__global__ void atomic_add_kernel(std::uint32_t* counters, std::uint32_t* checksums,
                                  std::uint32_t adds_per_thread, std::uint32_t slot_count,
                                  std::uint32_t thread_count, std::uint32_t seed) {
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t checksum = 0;
    for (std::uint32_t i = 0; i < adds_per_thread; ++i) {
        const std::uint32_t key = id + i * thread_count + seed;
        const std::uint32_t counter_index = hash_u32(key) % slot_count;
        checksum ^= atomicAdd(&counters[counter_index], 1u);
    }
    checksums[id] = checksum;
}

std::vector<Case> make_cases() {
    std::vector<Case> cases;
    cases.reserve((kMaxBufferExponent - kMinBufferExponent) * kStepsPerOctave + 1);
    for (std::uint32_t exponent = kMinBufferExponent; exponent < kMaxBufferExponent;
         ++exponent) {
        for (std::uint32_t step = 0; step < kStepsPerOctave; ++step) {
            const double log2_bytes =
                static_cast<double>(exponent) +
                static_cast<double>(step) / static_cast<double>(kStepsPerOctave);
            const auto slots = static_cast<std::uint64_t>(
                std::llround(std::exp2(log2_bytes) / sizeof(std::uint32_t)));
            cases.push_back(
                {slots * sizeof(std::uint32_t), kStepRelations[step], exponent});
        }
    }
    cases.push_back(
        {std::uint64_t{1} << kMaxBufferExponent, "power_of_two", kMaxBufferExponent});
    return cases;
}

double average(const std::vector<double>& values) {
    double sum = 0.0;
    for (const double value : values) {
        sum += value;
    }
    return sum / static_cast<double>(values.size());
}

double upper_median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

std::string cuda_version(int version) {
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

const char* os_name() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__linux__)
    return "Linux";
#elif defined(__APPLE__)
    return "macOS";
#else
    return "unknown";
#endif
}

std::string csv_field(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (const char character : value) {
        if (character == '"') {
            escaped += '"';
        }
        escaped += character;
    }
    escaped += '"';
    return escaped;
}

std::string nvidia_driver_version() {
#if defined(_WIN32)
    FILE* pipe = _popen(
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>NUL", "r");
#else
    FILE* pipe = popen(
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null", "r");
#endif
    if (pipe == nullptr) {
        return "unknown";
    }

    std::array<char, 128> output{};
    const char* read_result = std::fgets(output.data(), output.size(), pipe);
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    if (read_result == nullptr) {
        return "unknown";
    }

    std::string version = output.data();
    const std::size_t end = version.find_last_not_of(" \t\r\n");
    if (end == std::string::npos) {
        return "unknown";
    }
    version.erase(end + 1);
    return version;
}

void run(const Config& config) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (config.device >= static_cast<std::uint32_t>(device_count)) {
        throw std::runtime_error("--device " + std::to_string(config.device) +
                                 " is out of range; found " + std::to_string(device_count) +
                                 " CUDA device(s)");
    }
    CUDA_CHECK(cudaSetDevice(static_cast<int>(config.device)));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, static_cast<int>(config.device)));
    if (config.workgroups > static_cast<std::uint32_t>(properties.maxGridSize[0])) {
        throw std::runtime_error("--workgroups " + std::to_string(config.workgroups) +
                                 " exceeds this device's x grid limit (" +
                                 std::to_string(properties.maxGridSize[0]) + ")");
    }
    if (config.workgroups > std::numeric_limits<std::uint32_t>::max() / kBlockSize) {
        throw std::runtime_error("--workgroups makes the thread count exceed u32 range");
    }

    const std::uint32_t thread_count = config.workgroups * kBlockSize;
    if (config.adds_per_thread >
        std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(thread_count)) {
        throw std::runtime_error("requested workload exceeds u64 range");
    }
    const std::uint64_t total =
        static_cast<std::uint64_t>(thread_count) * config.adds_per_thread;
    const std::vector<Case> cases = make_cases();
    const std::uint64_t max_counter_size =
        std::max_element(cases.begin(), cases.end(), [](const Case& left, const Case& right) {
            return left.buffer_size < right.buffer_size;
        })->buffer_size;
    const std::uint64_t minimum_slots = cases.front().buffer_size / sizeof(std::uint32_t);
    if (total > minimum_slots * std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "workload can overflow individual counters at the smallest buffer size");
    }

    DeviceBuffer<std::uint32_t> counters(max_counter_size / sizeof(std::uint32_t));
    DeviceBuffer<std::uint32_t> checksums(thread_count);
    std::vector<std::uint32_t> readback(counters.count());
    Event gpu_start;
    Event gpu_stop;

    int cuda_driver_api_version = 0;
    int runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_api_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    const std::string driver_version = nvidia_driver_version();
    std::cerr << "GPU: " << properties.name << " (CUDA device " << config.device << ", sm_"
              << properties.major << properties.minor << ")\n";
    std::cerr << "OS/API: " << os_name() << " / CUDA " << cuda_version(runtime_version)
              << "\n";
    std::cerr << "NVIDIA driver: " << driver_version << "\n";
    std::cerr << "CUDA driver API/runtime: " << cuda_version(cuda_driver_api_version) << "/"
              << cuda_version(runtime_version) << "\n";
    std::cerr << "Sweep: " << cases.size()
              << " cases, " << config.samples
              << " samples each, 256 KiB through 64 MiB (ten log2 steps per octave), " << total
              << " atomic adds per sample\n";

    // Force lazy module loading/JIT compilation and warm up before collecting the sweep.
    CUDA_CHECK(cudaMemset(counters.get(), 0, counters.bytes()));
    atomic_add_kernel<<<config.workgroups, kBlockSize>>>(
        counters.get(), checksums.get(), config.adds_per_thread,
        static_cast<std::uint32_t>(counters.count()), thread_count, kSeed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Result> results;
    results.reserve(cases.size());
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const Case& benchmark_case = cases[case_index];
        const std::uint32_t slot_count =
            static_cast<std::uint32_t>(benchmark_case.buffer_size / sizeof(std::uint32_t));
        std::cerr << "[" << case_index + 1 << "/" << cases.size() << "] 2^"
                  << benchmark_case.exponent << " B " << benchmark_case.relation << ": "
                  << benchmark_case.buffer_size << " B (" << std::fixed << std::setprecision(6)
                  << static_cast<double>(benchmark_case.buffer_size) / (1024.0 * 1024.0)
                  << " MiB, " << slot_count << " u32 slots)\n";

        std::vector<double> host_throughputs;
        std::vector<double> gpu_throughputs;
        host_throughputs.reserve(config.samples);
        gpu_throughputs.reserve(config.samples);
        for (std::uint32_t sample = 1; sample <= config.samples; ++sample) {
            // Clearing and its synchronization are deliberately outside the timed region.
            CUDA_CHECK(cudaMemset(counters.get(), 0, benchmark_case.buffer_size));
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(gpu_start.get()));
            const auto host_start = std::chrono::steady_clock::now();
            atomic_add_kernel<<<config.workgroups, kBlockSize>>>(
                counters.get(), checksums.get(), config.adds_per_thread, slot_count,
                thread_count, kSeed);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(gpu_stop.get()));
            CUDA_CHECK(cudaEventSynchronize(gpu_stop.get()));
            const double host_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - host_start)
                    .count();
            float gpu_milliseconds = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&gpu_milliseconds, gpu_start.get(), gpu_stop.get()));
            const double gpu_seconds = static_cast<double>(gpu_milliseconds) / 1e3;
            const double host_throughput = static_cast<double>(total) / host_seconds / 1e9;
            const double gpu_throughput = static_cast<double>(total) / gpu_seconds / 1e9;
            std::cerr << "         sample " << sample << "/" << config.samples << ": host "
                      << std::fixed << std::setprecision(3) << host_seconds * 1e3 << " ms, "
                      << host_throughput << " Gatomic/s; GPU " << gpu_milliseconds << " ms, "
                      << gpu_throughput << " Gatomic/s\n";
            host_throughputs.push_back(host_throughput);
            gpu_throughputs.push_back(gpu_throughput);
        }

        CUDA_CHECK(cudaMemcpy(readback.data(), counters.get(), benchmark_case.buffer_size,
                              cudaMemcpyDeviceToHost));
        std::uint64_t actual = 0;
        for (std::uint32_t slot = 0; slot < slot_count; ++slot) {
            actual += readback[slot];
        }
        if (actual != total) {
            throw std::runtime_error("validation failed for " +
                                     std::to_string(benchmark_case.buffer_size) +
                                     " bytes: got sum " + std::to_string(actual) +
                                     ", expected " + std::to_string(total));
        }

        const double host_average = average(host_throughputs);
        const double host_median = upper_median(host_throughputs);
        const double gpu_average = average(gpu_throughputs);
        const double gpu_median = upper_median(gpu_throughputs);
        std::cerr << "         host average " << std::fixed << std::setprecision(3)
                  << host_average << ", median " << host_median << "; GPU average "
                  << gpu_average << ", median " << gpu_median
                  << " Gatomic/s; validation passed\n";
        results.push_back(
            {benchmark_case, host_average, host_median, gpu_average, gpu_median});
    }

    std::cerr << "Sweep complete; CSV follows on stdout.\n";
    std::cout << "buffer_bytes,buffer_mib,relation,power_of_two_exponent,"
                 "average_gatomic_per_s,median_gatomic_per_s,"
                 "average_gpu_gatomic_per_s,median_gpu_gatomic_per_s,"
                 "os,api,api_version,driver_version\n";
    for (const Result& result : results) {
        std::cout << result.benchmark_case.buffer_size << ',' << std::fixed
                  << std::setprecision(9)
                  << static_cast<double>(result.benchmark_case.buffer_size) /
                         (1024.0 * 1024.0)
                  << ',' << result.benchmark_case.relation << ','
                  << result.benchmark_case.exponent << ',' << std::setprecision(6)
                  << result.host_average << ',' << result.host_median << ','
                  << result.gpu_average << ',' << result.gpu_median << ','
                  << csv_field(os_name()) << ",CUDA," << cuda_version(runtime_version) << ','
                  << csv_field(driver_version) << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    Config config;
    try {
        config = parse_args(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        usage(argv[0]);
        return 2;
    }

    try {
        run(config);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
