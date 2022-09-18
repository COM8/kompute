// SPDX-License-Identifier: Apache-2.0

#include "kompute/Manager.hpp"
#include "fmt/format.h"
#include "kompute/Version.hpp"
#include "kompute/logger/Logger.hpp"
#include "vulkan/vulkan_enums.hpp"
#include "vulkan/vulkan_handles.hpp"
#include <cstddef>
#include <cstring>
#include <fmt/core.h>
#include <iterator>
#include <set>
#include <sstream>
#include <string>

namespace kp {

#ifndef KOMPUTE_DISABLE_VK_DEBUG_LAYERS
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugMessageCallback(VkDebugReportFlagsEXT /*flags*/,
                     VkDebugReportObjectTypeEXT /*objectType*/,
                     uint64_t /*object*/,
                     size_t /*location*/,
                     int32_t /*messageCode*/,
#if KOMPUTE_OPT_ACTIVE_LOG_LEVEL <= KOMPUTE_LOG_LEVEL_DEBUG
                     const char* pLayerPrefix,
                     const char* pMessage,
#else
                     const char* /*pLayerPrefix*/,
                     const char* /*pMessage*/,
#endif
                     void* /*pUserData*/)
{
    KP_LOG_DEBUG("[VALIDATION]: {} - {}", pLayerPrefix, pMessage);
    return VK_FALSE;
}

VkResult
CreateDebugUtilsMessengerEXT(
  vk::Instance& instance,
  const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
  const VkAllocationCallbacks* pAllocator,
  VkDebugUtilsMessengerEXT* pCallback)
{
    PFN_vkCreateDebugUtilsMessengerEXT func =
      reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        instance.getProcAddr("vkCreateDebugUtilsMessengerEXT"));
    if (func) {
        return func(instance, pCreateInfo, pAllocator, pCallback);
    }
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

VkBool32
debugUtilsMessageCallback(
  VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
  VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
  void* /*pUserData*/)
{
    KP_LOG_DEBUG("[VALIDATION][UTILS]: {}", pCallbackData->pMessage);
    return VK_FALSE;
}
#endif

Manager::Manager()
  : Manager(0)
{
}

Manager::Manager(uint32_t physicalDeviceIndex,
                 const std::vector<uint32_t>& familyQueueIndices,
                 const std::vector<std::string>& desiredExtensions)
{
    this->mManageResources = true;

// Make sure the logger is setup
#if !KOMPUTE_OPT_LOG_LEVEL_DISABLED
    logger::setupLogger();
#endif

    this->createInstance();
    this->createDevice(
      familyQueueIndices, physicalDeviceIndex, desiredExtensions);
}

Manager::Manager(std::shared_ptr<vk::Instance> instance,
                 std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                 std::shared_ptr<vk::Device> device)
{
    this->mManageResources = false;

    this->mInstance = instance;
    this->mPhysicalDevice = physicalDevice;
    this->mDevice = device;

// Make sure the logger is setup
#if !KOMPUTE_OPT_LOG_LEVEL_DISABLED
    logger::setupLogger();
#endif
}

Manager::~Manager()
{
    KP_LOG_DEBUG("Kompute Manager Destructor started");
    this->destroy();
}

void
Manager::destroy()
{

    KP_LOG_DEBUG("Kompute Manager destroy() started");

    if (this->mDevice == nullptr) {
        KP_LOG_ERROR(
          "Kompute Manager destructor reached with null Device pointer");
        return;
    }

    if (this->mManageResources && this->mManagedSequences.size()) {
        KP_LOG_DEBUG("Kompute Manager explicitly running destructor for "
                     "managed sequences");
        for (const std::weak_ptr<Sequence>& weakSq : this->mManagedSequences) {
            if (std::shared_ptr<Sequence> sq = weakSq.lock()) {
                sq->destroy();
            }
        }
        this->mManagedSequences.clear();
    }

    if (this->mManageResources && this->mManagedAlgorithms.size()) {
        KP_LOG_DEBUG("Kompute Manager explicitly freeing algorithms");
        for (const std::weak_ptr<Algorithm>& weakAlgorithm :
             this->mManagedAlgorithms) {
            if (std::shared_ptr<Algorithm> algorithm = weakAlgorithm.lock()) {
                algorithm->destroy();
            }
        }
        this->mManagedAlgorithms.clear();
    }

    if (this->mManageResources && this->mManagedTensors.size()) {
        KP_LOG_DEBUG("Kompute Manager explicitly freeing tensors");
        for (const std::weak_ptr<Tensor>& weakTensor : this->mManagedTensors) {
            if (std::shared_ptr<Tensor> tensor = weakTensor.lock()) {
                tensor->destroy();
            }
        }
        this->mManagedTensors.clear();
    }

    if (this->mFreeDevice) {
        KP_LOG_INFO("Destroying device");
        this->mDevice->destroy(
          (vk::Optional<const vk::AllocationCallbacks>)nullptr);
        this->mDevice = nullptr;
        KP_LOG_DEBUG("Kompute Manager Destroyed Device");
    }

    if (this->mInstance == nullptr) {
        KP_LOG_ERROR(
          "Kompute Manager destructor reached with null Instance pointer");
        return;
    }

#ifndef KOMPUTE_DISABLE_VK_DEBUG_LAYERS
    if (this->mDebugReportCallback) {
        this->mInstance->destroyDebugReportCallbackEXT(
          this->mDebugReportCallback, nullptr, this->mDebugDispatcher);
        KP_LOG_DEBUG("Kompute Manager Destroyed Debug Report Callback");
    }
    if (this->mDebugUtilsReportCallback) {
        this->mInstance->destroyDebugUtilsMessengerEXT(
          this->mDebugUtilsReportCallback);
        KP_LOG_DEBUG(
          "Kompute Manager Destroyed Debug Utils Messenger Callback");
    }
#endif

    if (this->mFreeInstance) {
        this->mInstance->destroy(
          (vk::Optional<const vk::AllocationCallbacks>)nullptr);
        this->mInstance = nullptr;
        KP_LOG_DEBUG("Kompute Manager Destroyed Instance");
    }
}

void
Manager::getIntersection(const std::vector<const char*>& v1,
                         const std::vector<const char*>& v2,
                         std::vector<const char*>& result)
{
    for (const char* s1 : v1) {
        for (const char* s2 : v2) {
            if (std::strcmp(s1, s2) == 0) {
                result.push_back(s1);
                break;
            }
        }
    }
}

void
Manager::createInstance()
{

    KP_LOG_DEBUG("Kompute Manager creating instance");

    this->mFreeInstance = true;

    vk::ApplicationInfo appInfo(
      "Kompute",
      VK_MAKE_VERSION(KP_VERSION_MAJOR, KP_VERSION_MINOR, KP_VERSION_MAJOR),
      "No Kompute Engine",
      VK_MAKE_VERSION(KP_VERSION_MAJOR, KP_VERSION_MINOR, KP_VERSION_MAJOR),
      KOMPUTE_VK_API_VERSION);

    // Enable extensions:
    std::vector<const char*> extRequested;
#ifndef KOMPUTE_DISABLE_VK_DEBUG_LAYERS
    extRequested.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    extRequested.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    // Check if all extensions are available:
    std::vector<vk::ExtensionProperties> availExts =
      vk::enumerateInstanceExtensionProperties();
    std::vector<const char*> availExtNames;
    availExtNames.reserve(availExts.size());
    for (const vk::ExtensionProperties& extProp : availExts) {
        availExtNames.push_back(extProp.extensionName);
    }
    KP_LOG_DEBUG(
      "Kompute Manager Available Vulkan extensions (amount:  {}): {}",
      availExtNames.size(),
      fmt::join(availExtNames, ", "));

    // Get the intersection between requested and available extensions:
    std::vector<const char*> extOverlap;
    getIntersection(extRequested, availExtNames, extOverlap);

    if (extOverlap.size() == extRequested.size()) {
        KP_LOG_INFO("Kompute Manager All ({}) requested Vulkan extensions got "
                    "enabled successfully.",
                    extRequested.size());
    } else {
        std::string err = fmt::format(
          "Kompute Manager Failed to create Vulkan instance! Only {} out of {} "
          "extensions are "
          "available.\nRequested extensions: {}\nAvailable extensions: {}",
          extOverlap.size(),
          extRequested.size(),
          fmt::join(extRequested, ", "),
          fmt::join(extOverlap, ", "));
        KP_LOG_ERROR("{}", err);
        throw std::runtime_error(err);
    }

#ifndef KOMPUTE_DISABLE_VK_DEBUG_LAYERS
    KP_LOG_DEBUG("Kompute Manager adding debug validation layers");
    // Enable validation layers:
    std::vector<const char*> layersRequested;

    // Get validation layer names from env variable:
    std::vector<std::string> envLayerNames;
    const char* envLayerNamesVal = std::getenv("KOMPUTE_ENV_DEBUG_LAYERS");
    if (envLayerNamesVal != nullptr && *envLayerNamesVal != '\0') {
        KP_LOG_DEBUG("Kompute Manager adding environment layers: {}",
                     envLayerNamesVal);
        std::istringstream iss(envLayerNamesVal);
        std::istream_iterator<std::string> beg(iss);
        std::istream_iterator<std::string> end;
        envLayerNames = std::vector<std::string>(beg, end);
        for (const std::string& layerName : envLayerNames) {
            layersRequested.push_back(layerName.c_str());
        }
        KP_LOG_DEBUG("Kompute Manager Desired layers: {}",
                     fmt::join(layersRequested, ", "));
    }

    // Check if all validation layers are available:
    std::vector<vk::LayerProperties> availLayers =
      vk::enumerateInstanceLayerProperties();
    std::vector<const char*> availLayerNames;
    availLayerNames.reserve(availLayers.size());
    for (const vk::LayerProperties& layerProp : availLayers) {
        availLayerNames.push_back(layerProp.layerName);
    }
    KP_LOG_DEBUG("Available Vulkan validation layers (amount: {}): {}",
                 availLayerNames.size(),
                 fmt::join(availLayerNames, ", "));

    // Get the intersection between requested and available validation layers:
    std::vector<const char*> layerOverlap;
    getIntersection(layersRequested, availLayerNames, layerOverlap);

    if (layerOverlap.size() == layersRequested.size()) {
        KP_LOG_INFO(
          "Kompute Manager All ({}) requested Vulkan validation layers "
          "got enabled successfully.",
          extRequested.size());
    } else {
        std::string err = fmt::format(
          "Kompute Manager Failed to create Vulkan instance! Only {} out of {} "
          "validation layers are available.\nRequested validation "
          "layers: {}\nAvailable validation layers: {}",
          layerOverlap.size(),
          layersRequested.size(),
          fmt::join(layersRequested, ", "),
          fmt::join(layerOverlap, ", "));
        KP_LOG_ERROR("{}", err);
        throw std::runtime_error(err);
    }
#endif

    vk::InstanceCreateInfo createInfo(
      {},
      &appInfo,
#ifndef KOMPUTE_DISABLE_VK_DEBUG_LAYERS
      layersRequested.size(),
      layersRequested.data(),
#else
      0,
      {}
#endif
      static_cast<uint32_t>(extRequested.size()),
      extRequested.data());

    std::vector<vk::ValidationFeatureEnableEXT> valFeaturesEnabled = {
        vk::ValidationFeatureEnableEXT::eDebugPrintf
    };
    vk::ValidationFeaturesEXT valFeatures(
      valFeaturesEnabled.size(), valFeaturesEnabled.data(), 0, nullptr);
    valFeatures.enabledValidationFeatureCount =
      static_cast<uint32_t>(valFeaturesEnabled.size());
    valFeatures.pEnabledValidationFeatures = valFeaturesEnabled.data();
    createInfo.setPNext(&valFeatures);

    this->mInstance = std::make_shared<vk::Instance>();
    vk::Result result =
      vk::createInstance(&createInfo, nullptr, this->mInstance.get());
    if (result == vk::Result::eSuccess) {
        KP_LOG_DEBUG("Kompute Manager Instance Created");
    } else {
        std::string err = fmt::format(
          "Kompute Manager Failed to create Vulkan instance! Results names can "
          "be found inside vulkan_enums.hpp. Result: {}",
          static_cast<int>(result));
        KP_LOG_ERROR("{}", err);
        throw std::runtime_error(err);
    }

#ifndef KOMPUTE_DISABLE_VK_DEBUG_LAYERS
    KP_LOG_DEBUG("Kompute Manager adding debug callbacks");
    vk::DebugReportFlagsEXT debugFlags =
      vk::DebugReportFlagBitsEXT::eError |
      vk::DebugReportFlagBitsEXT::eWarning |
      vk::DebugReportFlagBitsEXT::eDebug |
      vk::DebugReportFlagBitsEXT::eInformation |
      vk::DebugReportFlagBitsEXT::ePerformanceWarning;
    vk::DebugReportCallbackCreateInfoEXT debugCreateInfo = {};
    debugCreateInfo.pfnCallback =
      static_cast<PFN_vkDebugReportCallbackEXT>(debugMessageCallback);
    debugCreateInfo.flags = debugFlags;

    this->mDebugDispatcher.init(*this->mInstance, &vkGetInstanceProcAddr);
    this->mDebugReportCallback = this->mInstance->createDebugReportCallbackEXT(
      debugCreateInfo, nullptr, this->mDebugDispatcher);

    vk::DebugUtilsMessengerCreateInfoEXT debugInfoFlags(
      vk::DebugUtilsMessengerCreateFlagsEXT(),
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
      vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
      static_cast<PFN_vkDebugUtilsMessengerCallbackEXT>(
        debugUtilsMessageCallback),
      nullptr);

    if (CreateDebugUtilsMessengerEXT(
          *(this->mInstance),
          reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(
            &createInfo),
          nullptr,
          &this->mDebugUtilsReportCallback) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug callback!");
    }
#endif

    this->mInstance->createDebugUtilsMessengerEXT(debugInfoFlags);
}

void
Manager::clear()
{
    if (this->mManageResources) {
        this->mManagedTensors.erase(
          std::remove_if(begin(this->mManagedTensors),
                         end(this->mManagedTensors),
                         [](std::weak_ptr<Tensor> t) { return t.expired(); }),
          end(this->mManagedTensors));
        this->mManagedAlgorithms.erase(
          std::remove_if(
            begin(this->mManagedAlgorithms),
            end(this->mManagedAlgorithms),
            [](std::weak_ptr<Algorithm> t) { return t.expired(); }),
          end(this->mManagedAlgorithms));
        this->mManagedSequences.erase(
          std::remove_if(begin(this->mManagedSequences),
                         end(this->mManagedSequences),
                         [](std::weak_ptr<Sequence> t) { return t.expired(); }),
          end(this->mManagedSequences));
    }
}

void
Manager::createDevice(const std::vector<uint32_t>& familyQueueIndices,
                      uint32_t physicalDeviceIndex,
                      const std::vector<std::string>& desiredExtensions)
{

    KP_LOG_DEBUG("Kompute Manager creating Device");

    if (this->mInstance == nullptr) {
        throw std::runtime_error("Kompute Manager instance is null");
    }

    this->mFreeDevice = true;

    // Getting an integer that says how many vuklan devices we have
    std::vector<vk::PhysicalDevice> physicalDevices =
      this->mInstance->enumeratePhysicalDevices();
    uint32_t deviceCount = physicalDevices.size();

    // This means there are no devices at all
    if (deviceCount == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support! "
                                 "Maybe you haven't installed vulkan drivers?");
    }

    // This means that we're exceeding our device limit, for
    // example if we have 2 devices, just physicalDeviceIndex
    // 0 and 1 are acceptable. Hence, physicalDeviceIndex should
    // always be less than deviceCount, else we raise an error
    if (!(deviceCount > physicalDeviceIndex)) {
        throw std::runtime_error("There is no such physical index or device, "
                                 "please use your existing device");
    }

    vk::PhysicalDevice physicalDevice = physicalDevices[physicalDeviceIndex];

    this->mPhysicalDevice =
      std::make_shared<vk::PhysicalDevice>(physicalDevice);

#if KOMPUTE_OPT_ACTIVE_LOG_LEVEL <= KOMPUTE_LOG_LEVEL_INFO
    vk::PhysicalDeviceProperties physicalDeviceProperties =
      physicalDevice.getProperties();
#endif

    KP_LOG_INFO("Using physical device index {} found {}",
                physicalDeviceIndex,
                physicalDeviceProperties.deviceName);

    if (familyQueueIndices.empty()) {
        // Find compute queue
        std::vector<vk::QueueFamilyProperties> allQueueFamilyProperties =
          physicalDevice.getQueueFamilyProperties();

        uint32_t computeQueueFamilyIndex = 0;
        bool computeQueueSupported = false;
        for (uint32_t i = 0; i < allQueueFamilyProperties.size(); i++) {
            vk::QueueFamilyProperties queueFamilyProperties =
              allQueueFamilyProperties[i];

            if (queueFamilyProperties.queueFlags &
                vk::QueueFlagBits::eCompute) {
                computeQueueFamilyIndex = i;
                computeQueueSupported = true;
                break;
            }
        }

        if (!computeQueueSupported) {
            throw std::runtime_error("Compute queue is not supported");
        }

        this->mComputeQueueFamilyIndices.push_back(computeQueueFamilyIndex);
    } else {
        this->mComputeQueueFamilyIndices = familyQueueIndices;
    }

    std::unordered_map<uint32_t, uint32_t> familyQueueCounts;
    std::unordered_map<uint32_t, std::vector<float>> familyQueuePriorities;
    for (const auto& value : this->mComputeQueueFamilyIndices) {
        familyQueueCounts[value]++;
        familyQueuePriorities[value].push_back(1.0f);
    }

    std::unordered_map<uint32_t, uint32_t> familyQueueIndexCount;
    std::vector<vk::DeviceQueueCreateInfo> deviceQueueCreateInfos;
    for (const auto& familyQueueInfo : familyQueueCounts) {
        // Setting the device count to 0
        familyQueueIndexCount[familyQueueInfo.first] = 0;

        // Creating the respective device queue
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo(
          vk::DeviceQueueCreateFlags(),
          familyQueueInfo.first,
          familyQueueInfo.second,
          familyQueuePriorities[familyQueueInfo.first].data());
        deviceQueueCreateInfos.push_back(deviceQueueCreateInfo);
    }

    // Enable extensions:
    std::vector<const char*> extRequested;
    // Convert to cstring list internally to prevent having to change the API
    extRequested.reserve(desiredExtensions.size());
    for (const std::string& s : desiredExtensions) {
        extRequested.push_back(s.c_str());
    }

#ifndef KOMPUTE_DISABLE_VK_DEBUG_LAYERS
    // Allows printf debugging
    // https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/docs/debug_printf.md
    extRequested.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
#endif

    // Check if all extensions are available:
    std::vector<vk::ExtensionProperties> availExts =
      this->mPhysicalDevice->enumerateDeviceExtensionProperties();
    std::vector<const char*> availExtNames;
    availExtNames.reserve(availExts.size());
    for (const vk::ExtensionProperties& extProp : availExts) {
        availExtNames.push_back(extProp.extensionName);
    }
    KP_LOG_DEBUG(
      "Kompute Manager Available Vulkan device extensions (amount:  {}): {}",
      availExtNames.size(),
      fmt::join(availExtNames, ", "));

    // Get the intersection between requested and available extensions:
    std::vector<const char*> extOverlap;
    getIntersection(extRequested, availExtNames, extOverlap);

    if (extOverlap.size() == extRequested.size()) {
        KP_LOG_INFO(
          "Kompute Manager All ({}) requested Vulkan device extensions got "
          "enabled successfully.",
          extRequested.size());
    } else {
        std::string err = fmt::format(
          "Kompute Manager Failed to create Vulkan device! Only {} out of {} "
          "extensions are "
          "available.\nRequested extensions: {}\nAvailable extensions: {}",
          extOverlap.size(),
          extRequested.size(),
          fmt::join(extRequested, ", "),
          fmt::join(extOverlap, ", "));
        KP_LOG_ERROR("{}", err);
        throw std::runtime_error(err);
    }

#ifndef KOMPUTE_DISABLE_VK_DEBUG_LAYERS
    KP_LOG_DEBUG("Kompute Manager adding debug device validation layers");
    // Enable validation layers:
    std::vector<const char*> layersRequested;

    // Get validation layer names from env variable:
    std::vector<std::string> envLayerNames;
    const char* envLayerNamesVal =
      std::getenv("KOMPUTE_ENV_DEVICE_DEBUG_LAYERS");
    if (envLayerNamesVal != nullptr && *envLayerNamesVal != '\0') {
        KP_LOG_DEBUG("Kompute Manager adding device environment layers: {}",
                     envLayerNamesVal);
        std::istringstream iss(envLayerNamesVal);
        std::istream_iterator<std::string> beg(iss);
        std::istream_iterator<std::string> end;
        envLayerNames = std::vector<std::string>(beg, end);
        for (const std::string& layerName : envLayerNames) {
            layersRequested.push_back(layerName.c_str());
        }
        KP_LOG_DEBUG("Kompute Manager Desired device debug layers: {}",
                     fmt::join(layersRequested, ", "));
    }

    // Check if all validation layers are available:
    std::vector<vk::LayerProperties> availLayers =
      vk::enumerateInstanceLayerProperties();
    std::vector<const char*> availLayerNames;
    availLayerNames.reserve(availLayers.size());
    for (const vk::LayerProperties& layerProp : availLayers) {
        availLayerNames.push_back(layerProp.layerName);
    }
    KP_LOG_DEBUG("Available Vulkan device validation layers (amount: {}): {}",
                 availLayerNames.size(),
                 fmt::join(availLayerNames, ", "));

    // Get the intersection between requested and available validation layers:
    std::vector<const char*> layerOverlap;
    getIntersection(layersRequested, availLayerNames, layerOverlap);

    if (layerOverlap.size() == layersRequested.size()) {
        KP_LOG_INFO(
          "Kompute Manager All ({}) requested Vulkan device validation layers "
          "got enabled successfully.",
          layersRequested.size());
    } else {
        std::string err = fmt::format(
          "Kompute Manager Failed to create Vulkan device! Only {} out of {} "
          "validation layers are available.\nRequested validation "
          "layers: {}\nAvailable validation layers: {}",
          layerOverlap.size(),
          layersRequested.size(),
          fmt::join(layersRequested, ", "),
          fmt::join(layerOverlap, ", "));
        KP_LOG_ERROR("{}", err);
        throw std::runtime_error(err);
    }
#endif

    vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(),
                                          deviceQueueCreateInfos.size(),
                                          deviceQueueCreateInfos.data(),
#ifndef KOMPUTE_DISABLE_VK_DEBUG_LAYERS
                                          layersRequested.size(),
                                          layersRequested.data(),
#else
      0,
      {}
#endif
                                          extRequested.size(),
                                          extRequested.data());

    this->mDevice = std::make_shared<vk::Device>();
    physicalDevice.createDevice(
      &deviceCreateInfo, nullptr, this->mDevice.get());
    KP_LOG_DEBUG("Kompute Manager device created");

    for (const uint32_t& familyQueueIndex : this->mComputeQueueFamilyIndices) {
        std::shared_ptr<vk::Queue> currQueue = std::make_shared<vk::Queue>();

        this->mDevice->getQueue(familyQueueIndex,
                                familyQueueIndexCount[familyQueueIndex],
                                currQueue.get());

        familyQueueIndexCount[familyQueueIndex]++;

        this->mComputeQueues.push_back(currQueue);
    }

    KP_LOG_DEBUG("Kompute Manager compute queue obtained");
}

std::shared_ptr<Sequence>
Manager::sequence(uint32_t queueIndex, uint32_t totalTimestamps)
{
    KP_LOG_DEBUG("Kompute Manager sequence() with queueIndex: {}", queueIndex);

    std::shared_ptr<Sequence> sq{ new kp::Sequence(
      this->mPhysicalDevice,
      this->mDevice,
      this->mComputeQueues[queueIndex],
      this->mComputeQueueFamilyIndices[queueIndex],
      totalTimestamps) };

    if (this->mManageResources) {
        this->mManagedSequences.push_back(sq);
    }

    return sq;
}

vk::PhysicalDeviceProperties
Manager::getDeviceProperties() const
{
    return this->mPhysicalDevice->getProperties();
}

std::vector<vk::PhysicalDevice>
Manager::listDevices() const
{
    return this->mInstance->enumeratePhysicalDevices();
}

std::shared_ptr<vk::Instance>
Manager::getVkInstance() const
{
    return this->mInstance;
}

}
