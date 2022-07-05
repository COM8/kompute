// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "kompute/Kompute.hpp"

#include "shaders/Utils.hpp"

struct UniformBufferObject
{
    uint u1;
    uint u2;
    uint u3;
    uint u4;
} __attribute__((aligned(16)));

TEST(TestUniformBuffer, TestDestroyTensorSingle)
{
    kp::Manager mgr;

    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;

      layout(set = 0, binding = 0) buffer resultBuffer { uint result[]; };
      layout(set = 0, binding = 1) uniform uniformBufferObject {
        uint data[4];
      } ubo;

      void main() {
          uint index = gl_GlobalInvocationID.x;
          result[index] = ubo.data[0] + ubo.data[1] + ubo.data[2] + ubo.data[3];
      })");

    std::vector<uint32_t> spirv = compileSource(shader);

    // Result tensor:
    const size_t COUNT = 100000;
    std::vector<unsigned int> resultValues{};
    resultValues.resize(COUNT);
    for (size_t i = 0; i < COUNT; i++) {
        resultValues[i] = 0;
    }
    std::shared_ptr<kp::TensorT<unsigned int>> resultValuesTensor =
      mgr.tensorT(resultValues);

    // Data tensor:
    std::vector<unsigned int> data{ 1, 2, 3, 4 };
    std::shared_ptr<kp::TensorT<unsigned int>> dataTensor = mgr.tensorT(data);
    dataTensor->setDescriptorType(vk::DescriptorType::eUniformBuffer);

    std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm({ resultValuesTensor, dataTensor }, spirv);

    mgr.sequence()->eval<kp::OpTensorSyncDevice>(algo->getTensors());

    mgr.sequence()
      ->record<kp::OpAlgoDispatch>(algo)
      ->eval()
      ->eval<kp::OpTensorSyncLocal>(algo->getTensors());

    {
        std::shared_ptr<kp::Sequence> sq = nullptr;

        {

            tensorA = mgr.tensor(initialValues);

            std::shared_ptr<kp::Algorithm> algo =
              mgr.algorithm({ tensorA }, spirv);

            // Sync values to and from device
            mgr.sequence()->eval<kp::OpTensorSyncDevice>(algo->getTensors());

            EXPECT_EQ(tensorA->vector(), initialValues);

            mgr.sequence()
              ->record<kp::OpAlgoDispatch>(algo)
              ->eval()
              ->eval<kp::OpTensorSyncLocal>(algo->getTensors());

            const std::vector<float> expectedFinalValues = { 1.0f, 1.0f, 1.0f };
            EXPECT_EQ(tensorA->vector(), expectedFinalValues);

            tensorA->destroy();
            EXPECT_FALSE(tensorA->isInit());
        }
        EXPECT_FALSE(tensorA->isInit());
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
