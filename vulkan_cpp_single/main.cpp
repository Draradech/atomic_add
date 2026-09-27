#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Change these constants to select a different workload or Vulkan device.
constexpr std::uint32_t kDeviceIndex = 0;
constexpr std::uint32_t kWorkgroups = 4096;
constexpr std::uint32_t kWorkgroupSize = 256;  // Must match atomic_add.comp.
constexpr std::uint32_t kAddsPerThread = 1024;
constexpr std::uint32_t kSeed = 0x12345678u;
constexpr VkDeviceSize kCounterBytes = VkDeviceSize{8} * 1024 * 1024;
constexpr auto kReportInterval = std::chrono::seconds(5);
static_assert(kWorkgroups <= std::numeric_limits<std::uint32_t>::max() / kWorkgroupSize);

void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed (VkResult " +
                                 std::to_string(result) + ")");
    }
}

#define VK_CHECK(expression) check((expression), #expression)

struct Params {
    std::uint32_t adds_per_thread;
    std::uint32_t slot_count;
    std::uint32_t thread_count;
    std::uint32_t seed;
};
static_assert(sizeof(Params) == 16);

std::vector<std::uint32_t> read_shader() {
    std::ifstream file(ATOMIC_SHADER_PATH, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error(std::string("cannot open shader: ") + ATOMIC_SHADER_PATH);
    const auto length = file.tellg();
    if (length <= 0 || length % 4 != 0) throw std::runtime_error("invalid shader size");
    std::vector<std::uint32_t> words(static_cast<std::size_t>(length / 4));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), length);
    if (!file) throw std::runtime_error("cannot read shader");
    return words;
}

class VulkanRunner {
public:
    VulkanRunner() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "vulkan-atomic-add-single";
        app.apiVersion = VK_API_VERSION_1_0;
        VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instance_info.pApplicationInfo = &app;
        VK_CHECK(vkCreateInstance(&instance_info, nullptr, &instance_));

        std::uint32_t device_count = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &device_count, nullptr));
        if (kDeviceIndex >= device_count) throw std::runtime_error("Vulkan device index is out of range");
        std::vector<VkPhysicalDevice> devices(device_count);
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()));
        physical_ = devices[kDeviceIndex];

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical_, &properties);
        std::cerr << "GPU: " << properties.deviceName << '\n';
        const auto& limits = properties.limits;
        if (kWorkgroups > limits.maxComputeWorkGroupCount[0] ||
            kWorkgroupSize > limits.maxComputeWorkGroupSize[0] ||
            kWorkgroupSize > limits.maxComputeWorkGroupInvocations ||
            kCounterBytes > limits.maxStorageBufferRange ||
            checksum_bytes() > limits.maxStorageBufferRange ||
            sizeof(Params) > limits.maxUniformBufferRange) {
            throw std::runtime_error("workload exceeds Vulkan device limits");
        }

        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, families.data());
        std::uint32_t queue_family = family_count;
        for (std::uint32_t i = 0; i < family_count; ++i) {
            if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                queue_family = i;
                break;
            }
        }
        if (queue_family == family_count) throw std::runtime_error("device has no compute queue");

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        VK_CHECK(vkCreateDevice(physical_, &device_info, nullptr, &device_));
        vkGetDeviceQueue(device_, queue_family, 0, &queue_);
        vkGetPhysicalDeviceMemoryProperties(physical_, &memory_properties_);

        make_buffer(counters_, kCounterBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        make_buffer(checksums_, checksum_bytes(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        make_buffer(params_, sizeof(Params), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* mapped = nullptr;
        VK_CHECK(vkMapMemory(device_, params_.memory, 0, sizeof(Params), 0, &mapped));
        const Params values{kAddsPerThread, static_cast<std::uint32_t>(kCounterBytes / 4),
                            kWorkgroups * kWorkgroupSize, kSeed};
        std::memcpy(mapped, &values, sizeof(values));
        vkUnmapMemory(device_, params_.memory);

        const VkDescriptorSetLayoutBinding bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = 3;
        layout_info.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_layout_));
        const VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        };
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 2;
        pool_info.pPoolSizes = pool_sizes;
        VK_CHECK(vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_));
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = descriptor_pool_;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &descriptor_layout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &set_info, &descriptor_set_));

        const VkDescriptorBufferInfo buffer_infos[] = {
            {counters_.buffer, 0, kCounterBytes},
            {params_.buffer, 0, sizeof(Params)},
            {checksums_.buffer, 0, checksum_bytes()},
        };
        VkWriteDescriptorSet writes[3]{};
        for (std::uint32_t i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set_;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = i == 1 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                             : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_infos[i];
        }
        vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

        VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_layout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_));
        const auto shader_code = read_shader();
        VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader_info.codeSize = shader_code.size() * sizeof(std::uint32_t);
        shader_info.pCode = shader_code.data();
        VK_CHECK(vkCreateShaderModule(device_, &shader_info, nullptr, &shader_));
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = shader_;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = pipeline_layout_;
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                          &pipeline_));

        VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        command_pool_info.queueFamilyIndex = queue_family;
        VK_CHECK(vkCreateCommandPool(device_, &command_pool_info, nullptr, &command_pool_));
        VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_info.commandPool = command_pool_;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device_, &command_info, &command_));
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(command_, &begin_info));
        vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                                0, 1, &descriptor_set_, 0, nullptr);
        vkCmdDispatch(command_, kWorkgroups, 1, 1);
        VK_CHECK(vkEndCommandBuffer(command_));

        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(device_, &fence_info, nullptr, &fence_));
    }

    VulkanRunner(const VulkanRunner&) = delete;
    VulkanRunner& operator=(const VulkanRunner&) = delete;

    ~VulkanRunner() {
        if (device_) {
            vkDeviceWaitIdle(device_);
            if (fence_) vkDestroyFence(device_, fence_, nullptr);
            if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
            if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
            if (shader_) vkDestroyShaderModule(device_, shader_, nullptr);
            if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
            if (descriptor_pool_) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
            if (descriptor_layout_) vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
            destroy_buffer(params_);
            destroy_buffer(checksums_);
            destroy_buffer(counters_);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    void dispatch() {
        VK_CHECK(vkResetFences(device_, 1, &fence_));
        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_;
        VK_CHECK(vkQueueSubmit(queue_, 1, &submit_info, fence_));
        VK_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));
    }

private:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };

    static constexpr VkDeviceSize checksum_bytes() {
        return VkDeviceSize{kWorkgroups} * kWorkgroupSize * sizeof(std::uint32_t);
    }

    void make_buffer(Buffer& buffer, VkDeviceSize bytes, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags memory_flags) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_, &info, nullptr, &buffer.buffer));
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, buffer.buffer, &requirements);
        std::uint32_t type = memory_properties_.memoryTypeCount;
        for (std::uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memory_properties_.memoryTypes[i].propertyFlags & memory_flags) == memory_flags) {
                type = i;
                break;
            }
        }
        if (type == memory_properties_.memoryTypeCount)
            throw std::runtime_error("no compatible Vulkan memory type");
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        VK_CHECK(vkAllocateMemory(device_, &allocation, nullptr, &buffer.memory));
        VK_CHECK(vkBindBufferMemory(device_, buffer.buffer, buffer.memory, 0));
    }

    void destroy_buffer(Buffer& buffer) {
        if (buffer.buffer) vkDestroyBuffer(device_, buffer.buffer, nullptr);
        if (buffer.memory) vkFreeMemory(device_, buffer.memory, nullptr);
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_properties_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    Buffer counters_;
    Buffer checksums_;
    Buffer params_;
    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkShaderModule shader_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

}  // namespace

int main() {
    try {
        VulkanRunner runner;
        constexpr double adds_per_dispatch =
            static_cast<double>(kWorkgroups) * kWorkgroupSize * kAddsPerThread;
        std::cerr << "Running 8 MiB workload continuously; stop with Ctrl-C.\n";
        auto interval_start = std::chrono::steady_clock::now();
        std::uint64_t dispatches = 0;
        for (;;) {
            runner.dispatch();
            ++dispatches;
            const auto now = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(now - interval_start).count();
            if (now - interval_start >= kReportInterval) {
                std::cout << std::fixed << std::setprecision(3)
                          << dispatches * adds_per_dispatch / seconds / 1e9
                          << " Gatomicadds/s" << std::endl;
                dispatches = 0;
                interval_start = std::chrono::steady_clock::now();
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
