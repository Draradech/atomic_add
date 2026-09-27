#include <vulkan/vulkan.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kWorkgroupSize = 256;
constexpr std::uint32_t kMinBufferExponent = 18;
constexpr std::uint32_t kMaxBufferExponent = 26;
constexpr std::uint32_t kStepsPerOctave = 10;
constexpr std::uint32_t kSeed = 0x12345678u;
constexpr VkDeviceSize kMinCounterBytes = VkDeviceSize{1} << kMinBufferExponent;
constexpr VkDeviceSize kMaxCounterBytes = VkDeviceSize{1} << kMaxBufferExponent;
constexpr VkBufferUsageFlags kCounterUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;

void vk_check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed (VkResult " +
                                 std::to_string(result) + ")");
    }
}

#define VK_CHECK(expression) vk_check((expression), #expression)

struct Config {
    std::uint32_t device = 0;
    std::uint32_t workgroups = 4096;
    std::uint32_t adds_per_thread = 1024;
    std::uint32_t samples = 7;
    bool small_buffers = false;
};

struct Result {
    std::uint64_t buffer_size;
    double median;
};

void usage(const char* program) {
    std::cerr << "Usage: " << program
              << " [--device N] [--workgroups N] [--adds N] [--samples N]"
                 " [--small-buffers]\n"
                 "Defaults: --device 0 --workgroups 4096 --adds 1024 --samples 7; "
                 "one 64 MiB counter buffer\n";
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
        if (flag == "--small-buffers") {
            config.small_buffers = true;
            continue;
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

std::vector<std::uint64_t> buffer_sizes() {
    std::vector<std::uint64_t> sizes;
    sizes.reserve((kMaxBufferExponent - kMinBufferExponent) * kStepsPerOctave + 1);
    for (std::uint32_t exponent = kMinBufferExponent; exponent < kMaxBufferExponent; ++exponent) {
        for (std::uint32_t step = 0; step < kStepsPerOctave; ++step) {
            const double log2_bytes = static_cast<double>(exponent) +
                                      static_cast<double>(step) / kStepsPerOctave;
            const auto slots = static_cast<std::uint64_t>(
                std::llround(std::exp2(log2_bytes) / sizeof(std::uint32_t)));
            sizes.push_back(slots * sizeof(std::uint32_t));
        }
    }
    sizes.push_back(kMaxCounterBytes);
    return sizes;
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

std::string metadata_value(std::string value) {
    std::replace(value.begin(), value.end(), '\r', ' ');
    std::replace(value.begin(), value.end(), '\n', ' ');
    return value;
}

std::string vk_version(std::uint32_t version) {
    return std::to_string(VK_VERSION_MAJOR(version)) + "." +
           std::to_string(VK_VERSION_MINOR(version));
}

std::vector<std::uint32_t> read_shader() {
    std::ifstream file(ATOMIC_SHADER_PATH, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error(std::string("cannot open shader: ") + ATOMIC_SHADER_PATH);
    }
    const std::streamoff length = file.tellg();
    if (length <= 0 || length % 4 != 0) {
        throw std::runtime_error("shader size is not a nonzero multiple of four bytes");
    }
    std::vector<std::uint32_t> words(static_cast<std::size_t>(length / 4));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), length);
    if (!file) {
        throw std::runtime_error("cannot read compiled shader");
    }
    return words;
}

class VulkanBench {
public:
    VulkanBench() = default;
    VulkanBench(const VulkanBench&) = delete;
    VulkanBench& operator=(const VulkanBench&) = delete;

    ~VulkanBench() {
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
            if (readback_.mapped) vkUnmapMemory(device_, readback_.memory);
            if (params_.mapped) vkUnmapMemory(device_, params_.memory);
            if (fence_) vkDestroyFence(device_, fence_, nullptr);
            if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
            if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
            if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
            if (descriptor_pool_) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
            if (descriptor_layout_) vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
            destroy_buffer(readback_);
            destroy_buffer(params_);
            destroy_buffer(checksums_);
            destroy_buffer(counters_);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    void initialize(const Config& config, std::uint64_t total) {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "atomic-add-bench";
        app.apiVersion = VK_API_VERSION_1_0;
        const auto enumerate_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
        if (enumerate_version) {
            std::uint32_t loader_version = 0;
            VK_CHECK(enumerate_version(&loader_version));
            if (loader_version >= VK_API_VERSION_1_2) app.apiVersion = VK_API_VERSION_1_2;
        }
        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app;
        VK_CHECK(vkCreateInstance(&instance_info, nullptr, &instance_));

        std::uint32_t device_count = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &device_count, nullptr));
        if (config.device >= device_count) {
            throw std::runtime_error("--device " + std::to_string(config.device) +
                                     " is out of range; found " + std::to_string(device_count) +
                                     " Vulkan device(s)");
        }
        std::vector<VkPhysicalDevice> devices(device_count);
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()));
        physical_ = devices[config.device];
        vkGetPhysicalDeviceProperties(physical_, &properties_);
        vkGetPhysicalDeviceMemoryProperties(physical_, &memory_properties_);
        if (app.apiVersion >= VK_API_VERSION_1_2 &&
            properties_.apiVersion >= VK_API_VERSION_1_2) {
            VkPhysicalDeviceDriverProperties driver{};
            driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
            VkPhysicalDeviceProperties2 extended{};
            extended.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            extended.pNext = &driver;
            vkGetPhysicalDeviceProperties2(physical_, &extended);
            if (driver.driverName[0]) {
                driver_label_ = driver.driverName;
                if (driver.driverInfo[0]) driver_label_ += std::string(" ") + driver.driverInfo;
            }
        }

        const auto& limits = properties_.limits;
        if (config.workgroups > limits.maxComputeWorkGroupCount[0]) {
            throw std::runtime_error("--workgroups exceeds this device's x dispatch limit (" +
                                     std::to_string(limits.maxComputeWorkGroupCount[0]) + ")");
        }
        if (limits.maxComputeWorkGroupSize[0] < kWorkgroupSize ||
            limits.maxComputeWorkGroupInvocations < kWorkgroupSize) {
            throw std::runtime_error("device does not support 256-thread compute workgroups");
        }
        if (config.workgroups > std::numeric_limits<std::uint32_t>::max() / kWorkgroupSize) {
            throw std::runtime_error("--workgroups makes the thread count exceed u32 range");
        }
        if (kMaxCounterBytes > limits.maxStorageBufferRange) {
            throw std::runtime_error("64 MiB counter buffer exceeds maxStorageBufferRange");
        }
        const VkDeviceSize checksum_bytes =
            static_cast<VkDeviceSize>(config.workgroups) * kWorkgroupSize * sizeof(std::uint32_t);
        if (checksum_bytes > limits.maxStorageBufferRange) {
            throw std::runtime_error("checksum buffer exceeds maxStorageBufferRange");
        }
        if (sizeof(Params) > limits.maxUniformBufferRange) {
            throw std::runtime_error("parameter block exceeds maxUniformBufferRange");
        }
        if (total > (kMaxCounterBytes / sizeof(std::uint32_t)) *
                        std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("workload can overflow an individual counter");
        }

        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, families.data());
        queue_family_ = family_count;
        for (std::uint32_t index = 0; index < family_count; ++index) {
            if (families[index].queueCount && (families[index].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                queue_family_ = index;
                if (!(families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;
            }
        }
        if (queue_family_ == family_count) {
            throw std::runtime_error("device has no compute queue");
        }
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family_;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        VK_CHECK(vkCreateDevice(physical_, &device_info, nullptr, &device_));
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

        counter_buffer_bytes_ = config.small_buffers ? kMinCounterBytes : kMaxCounterBytes;
        counters_ = create_buffer(counter_buffer_bytes_, kCounterUsage, 0,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        checksums_ = create_buffer(checksum_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        params_ = create_buffer(sizeof(Params), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        readback_ = create_buffer(kMaxCounterBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkMapMemory(device_, params_.memory, 0, VK_WHOLE_SIZE, 0, &params_.mapped));
        VK_CHECK(vkMapMemory(device_, readback_.memory, 0, VK_WHOLE_SIZE, 0, &readback_.mapped));

        create_pipeline();
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = queue_family_;
        VK_CHECK(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_));
        VkCommandBufferAllocateInfo command_info{};
        command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_info.commandPool = command_pool_;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device_, &command_info, &commands_));
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_CHECK(vkCreateFence(device_, &fence_info, nullptr, &fence_));
    }

    const VkPhysicalDeviceProperties& properties() const { return properties_; }

    std::string driver_name() const {
        if (!driver_label_.empty()) return driver_label_;
        if (properties_.vendorID == 0x10de) {
            const std::uint32_t version = properties_.driverVersion;
            const auto patch = (version >> 6) & 0xff;
            return "NVIDIA " + std::to_string(version >> 22) + "." +
                   std::to_string((version >> 14) & 0xff) + "." +
                   (patch < 10 ? "0" : "") + std::to_string(patch);
        }
        return "Vulkan driver " + std::to_string(properties_.driverVersion);
    }

    void set_params(std::uint32_t adds, std::uint32_t slots, std::uint32_t threads) {
        const Params values{adds, slots, threads, kSeed};
        std::memcpy(params_.mapped, &values, sizeof(values));
        if (!(params_.memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            VkMappedMemoryRange range{};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = params_.memory;
            range.size = VK_WHOLE_SIZE;
            VK_CHECK(vkFlushMappedMemoryRanges(device_, 1, &range));
        }
    }

    void replace_counter_buffer(VkDeviceSize bytes) {
        destroy_buffer(counters_);
        counters_ = {};
        counter_buffer_bytes_ = 0;
        counters_ = create_buffer(bytes, kCounterUsage, 0,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        counter_buffer_bytes_ = bytes;
    }

    void set_counter_range(VkDeviceSize bytes) {
        const VkDescriptorBufferInfo info{counters_.buffer, 0, bytes};
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor_set_;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &info;
        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    }

    void clear(VkDeviceSize bytes) {
        submit([&] {
            vkCmdFillBuffer(commands_, counters_.buffer, 0, bytes, 0);
            buffer_barrier(counters_.buffer, bytes, VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        });
    }

    double dispatch(std::uint32_t workgroups) {
        return submit([&] {
            vkCmdBindPipeline(commands_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
            vkCmdBindDescriptorSets(commands_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                                    0, 1, &descriptor_set_, 0, nullptr);
            vkCmdDispatch(commands_, workgroups, 1, 1);
            buffer_barrier(counters_.buffer, counter_buffer_bytes_, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT);
        });
    }

    std::uint64_t sum_counters(VkDeviceSize bytes) {
        submit([&] {
            VkBufferCopy region{0, 0, bytes};
            vkCmdCopyBuffer(commands_, counters_.buffer, readback_.buffer, 1, &region);
            buffer_barrier(readback_.buffer, bytes, VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT);
        });
        if (!(readback_.memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            VkMappedMemoryRange range{};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = readback_.memory;
            range.size = VK_WHOLE_SIZE;
            VK_CHECK(vkInvalidateMappedMemoryRanges(device_, 1, &range));
        }
        const auto* words = static_cast<const std::uint32_t*>(readback_.mapped);
        std::uint64_t sum = 0;
        for (VkDeviceSize index = 0; index < bytes / sizeof(std::uint32_t); ++index) {
            sum += words[index];
        }
        return sum;
    }

private:
    struct Params {
        std::uint32_t adds_per_thread;
        std::uint32_t slot_count;
        std::uint32_t thread_count;
        std::uint32_t seed;
    };
    static_assert(sizeof(Params) == 16);

    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkMemoryPropertyFlags memory_flags = 0;
        void* mapped = nullptr;
    };

    void destroy_buffer(Buffer& buffer) {
        if (buffer.buffer) vkDestroyBuffer(device_, buffer.buffer, nullptr);
        if (buffer.memory) vkFreeMemory(device_, buffer.memory, nullptr);
    }

    Buffer create_buffer(VkDeviceSize bytes, VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) {
        Buffer result;
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = bytes;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_, &info, nullptr, &result.buffer));
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, result.buffer, &requirements);
        std::uint32_t memory_type = memory_properties_.memoryTypeCount;
        for (std::uint32_t index = 0; index < memory_properties_.memoryTypeCount; ++index) {
            const auto flags = memory_properties_.memoryTypes[index].propertyFlags;
            if ((requirements.memoryTypeBits & (1u << index)) && (flags & required) == required) {
                if (memory_type == memory_properties_.memoryTypeCount) memory_type = index;
                if ((flags & preferred) == preferred) {
                    memory_type = index;
                    break;
                }
            }
        }
        if (memory_type == memory_properties_.memoryTypeCount) {
            vkDestroyBuffer(device_, result.buffer, nullptr);
            throw std::runtime_error("no compatible Vulkan memory type for buffer");
        }
        result.memory_flags = memory_properties_.memoryTypes[memory_type].propertyFlags;
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memory_type;
        VkResult status = vkAllocateMemory(device_, &allocation, nullptr, &result.memory);
        if (status != VK_SUCCESS) {
            vkDestroyBuffer(device_, result.buffer, nullptr);
            vk_check(status, "vkAllocateMemory");
        }
        status = vkBindBufferMemory(device_, result.buffer, result.memory, 0);
        if (status != VK_SUCCESS) {
            vkDestroyBuffer(device_, result.buffer, nullptr);
            vkFreeMemory(device_, result.memory, nullptr);
            vk_check(status, "vkBindBufferMemory");
        }
        return result;
    }

    void create_pipeline() {
        const VkDescriptorSetLayoutBinding bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = 3;
        layout_info.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_layout_));
        const VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        };
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 2;
        pool_info.pPoolSizes = pool_sizes;
        VK_CHECK(vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_));
        VkDescriptorSetAllocateInfo set_info{};
        set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        set_info.descriptorPool = descriptor_pool_;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &descriptor_layout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &set_info, &descriptor_set_));
        const VkDescriptorBufferInfo param_info{params_.buffer, 0, sizeof(Params)};
        const VkDescriptorBufferInfo checksum_info{checksums_.buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet writes[2]{};
        for (auto& write : writes) {
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptor_set_;
            write.descriptorCount = 1;
        }
        writes[0].dstBinding = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &param_info;
        writes[1].dstBinding = 2;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &checksum_info;
        vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);

        VkPipelineLayoutCreateInfo pipeline_layout_info{};

        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_layout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_));
        const auto shader = read_shader();
        VkShaderModuleCreateInfo shader_info{};
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = shader.size() * sizeof(std::uint32_t);
        shader_info.pCode = shader.data();
        VkShaderModule module = VK_NULL_HANDLE;
        VK_CHECK(vkCreateShaderModule(device_, &shader_info, nullptr, &module));
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage;
        pipeline_info.layout = pipeline_layout_;
        const VkResult status = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1,
                                                           &pipeline_info, nullptr, &pipeline_);
        vkDestroyShaderModule(device_, module, nullptr);
        vk_check(status, "vkCreateComputePipelines");
    }

    void buffer_barrier(VkBuffer buffer, VkDeviceSize bytes, VkAccessFlags source_access,
                        VkAccessFlags destination_access, VkPipelineStageFlags source_stage,
                        VkPipelineStageFlags destination_stage) {
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = source_access;
        barrier.dstAccessMask = destination_access;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = buffer;
        barrier.offset = 0;
        barrier.size = bytes;
        vkCmdPipelineBarrier(commands_, source_stage, destination_stage, 0, 0, nullptr,
                             1, &barrier, 0, nullptr);
    }

    template <typename Record>
    double submit(Record record) {
        VK_CHECK(vkResetCommandBuffer(commands_, 0));
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(commands_, &begin));
        record();
        VK_CHECK(vkEndCommandBuffer(commands_));
        VkSubmitInfo submission{};
        submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submission.commandBufferCount = 1;
        submission.pCommandBuffers = &commands_;
        VK_CHECK(vkResetFences(device_, 1, &fence_));
        const auto start = std::chrono::steady_clock::now();
        VK_CHECK(vkQueueSubmit(queue_, 1, &submission, fence_));
        VK_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties_{};
    VkPhysicalDeviceMemoryProperties memory_properties_{};
    std::string driver_label_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    Buffer counters_;
    VkDeviceSize counter_buffer_bytes_ = 0;
    Buffer checksums_;
    Buffer params_;
    Buffer readback_;
    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer commands_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

void run(const Config& config) {
    if (config.workgroups > std::numeric_limits<std::uint32_t>::max() / kWorkgroupSize) {
        throw std::runtime_error("--workgroups makes the thread count exceed u32 range");
    }
    const std::uint32_t thread_count = config.workgroups * kWorkgroupSize;
    const std::uint64_t total = static_cast<std::uint64_t>(thread_count) * config.adds_per_thread;
    const auto sizes = buffer_sizes();
    if (total > (sizes.front() / sizeof(std::uint32_t)) *
                    std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "workload can overflow individual counters at the smallest buffer size");
    }

    VulkanBench bench;
    bench.initialize(config, total);
    const auto& properties = bench.properties();
    std::cerr << "GPU: " << properties.deviceName << " (Vulkan device " << config.device << ")\n";
    std::cerr << "Driver: " << bench.driver_name() << "\n";
    std::cerr << "Sweep: " << sizes.size() << " cases, " << config.samples
              << " samples each, 256 KiB through 64 MiB (ten log2 steps per octave), "
              << total << " atomic adds per sample\n";
    if (config.small_buffers) {
        std::cerr << "Counter buffer: allocated separately at each sweep size\n";
    }

    const VkDeviceSize warmup_bytes = config.small_buffers ? kMinCounterBytes : kMaxCounterBytes;
    bench.set_params(config.adds_per_thread,
                     static_cast<std::uint32_t>(warmup_bytes / sizeof(std::uint32_t)),
                     thread_count);
    bench.set_counter_range(warmup_bytes);
    bench.clear(warmup_bytes);
    bench.dispatch(config.workgroups);

    std::vector<Result> results;
    results.reserve(sizes.size());
    for (std::size_t case_index = 0; case_index < sizes.size(); ++case_index) {
        const auto buffer_size = sizes[case_index];
        const auto slot_count = static_cast<std::uint32_t>(buffer_size / sizeof(std::uint32_t));
        std::cerr << "[" << case_index + 1 << "/" << sizes.size() << "] " << std::fixed
                  << std::setprecision(6)
                  << static_cast<double>(buffer_size) / (1024.0 * 1024.0) << " MiB ("
                  << slot_count << " u32 slots)\n";
        if (config.small_buffers && case_index != 0) {
            bench.replace_counter_buffer(buffer_size);
        }
        bench.set_params(config.adds_per_thread, slot_count, thread_count);
        bench.set_counter_range(buffer_size);
        std::vector<double> throughputs;
        throughputs.reserve(config.samples);
        for (std::uint32_t sample = 1; sample <= config.samples; ++sample) {
            bench.clear(buffer_size);
            const double seconds = bench.dispatch(config.workgroups);
            const double throughput = static_cast<double>(total) / seconds / 1e9;
            std::cerr << "         sample " << sample << "/" << config.samples << ": "
                      << std::fixed << std::setprecision(3) << seconds * 1e3 << " ms, "
                      << throughput << " Gatomic/s\n";
            throughputs.push_back(throughput);
        }
        const std::uint64_t actual = bench.sum_counters(buffer_size);
        if (actual != total) {
            throw std::runtime_error("validation failed for " + std::to_string(buffer_size) +
                                     " bytes: got sum " + std::to_string(actual) +
                                     ", expected " + std::to_string(total));
        }
        std::sort(throughputs.begin(), throughputs.end());
        const double median = throughputs[throughputs.size() / 2];
        std::cerr << "         median " << std::fixed << std::setprecision(3) << median
                  << " Gatomic/s; validation passed\n";
        results.push_back({buffer_size, median});
    }

    std::cerr << "Sweep complete; CSV follows on stdout.\n";
    std::cout << "# environment=" << os_name() << '\n';
    std::cout << "# api=Vulkan " << vk_version(properties.apiVersion) << '\n';
    std::cout << "# device=" << metadata_value(properties.deviceName) << '\n';
    std::cout << "# driver=" << metadata_value(bench.driver_name()) << '\n';
    if (config.small_buffers) std::cout << "# counter_allocation=per-step\n";
    std::cout << "buffer_bytes,median_gatomic_per_s\n";
    for (const Result& result : results) {
        std::cout << result.buffer_size << ',' << std::fixed << std::setprecision(6)
                  << result.median << '\n';
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
