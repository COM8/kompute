// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "kompute/Kompute.hpp"

#include "kompute/Tensor.hpp"
#include "shaders/Utils.hpp"

TEST(TestUniformBuffer, TestUniformBufferSum)
{
    kp::Manager mgr;

    std::string shader(R"(
      #version 450

      // Ensure we have a compact layout for uniform arrays.
      // Else we would have to pass multiples of sizeof(vec4) when binding.
      // Source: https://www.reddit.com/r/vulkan/comments/u5jiws/comment/i575o3i/?utm_source=share&utm_medium=web2x&context=3
      #extension GL_EXT_scalar_block_layout : require

      layout (local_size_x = 1) in;

      layout(set = 0, binding = 0) buffer resultBuffer { uint result[]; };
      layout(set = 0, binding = 1, std430) uniform uniformBufferObject {
        uint data[4];
      };

      void main() {
          uint index = gl_GlobalInvocationID.x;
          result[index] = data[0] + data[1] + data[2] + data[3];
      })");

    std::vector<uint32_t> spirv = compileSource(shader);

    // Result tensor:
    const size_t COUNT = 2;
    std::vector<unsigned int> resultValues{};
    resultValues.resize(COUNT);
    for (size_t i = 0; i < COUNT; i++) {
        resultValues[i] = 0;
    }
    std::shared_ptr<kp::TensorT<unsigned int>> resultValuesTensor =
      mgr.tensorT(resultValues);

    // Data tensor:
    std::vector<unsigned int> data{ 3, 4, 5, 6 };
    std::shared_ptr<kp::Tensor> dataTensor =
      mgr.tensor(data.data(),
                 data.size(),
                 sizeof(unsigned int),
                 kp::Tensor::TensorDataTypes::eUnsignedInt);
    dataTensor->setDescriptorType(vk::DescriptorType::eUniformBuffer);

    std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm({ resultValuesTensor, dataTensor }, spirv);

    mgr.sequence()->eval<kp::OpTensorSyncDevice>(algo->getTensors());

    mgr.sequence()
      ->record<kp::OpAlgoDispatch>(algo)
      ->eval()
      ->eval<kp::OpTensorSyncLocal>(algo->getTensors());

    std::vector<unsigned int> results = resultValuesTensor->vector();
    for (unsigned int result : results) {
        EXPECT_EQ(result, 18);
    }
}

int
main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);

#if KOMPUTE_ENABLE_SPDLOG
    spdlog::set_level(
      static_cast<spdlog::level::level_enum>(KOMPUTE_LOG_LEVEL));
#endif

    return RUN_ALL_TESTS();
}
