#pragma once

#include "kompute/Algorithm.hpp"
#include "kompute/Core.hpp"
#include "kompute/Manager.hpp"
#include "kompute/Sequence.hpp"
#include "kompute/Tensor.hpp"

#include "kompute/operations/OpAlgoDispatch.hpp"
#include "kompute/operations/OpBase.hpp"
#include "kompute/operations/OpMemoryBarrier.hpp"
#include "kompute/operations/OpMult.hpp"
#include "kompute/operations/OpTensorCopy.hpp"
#include "kompute/operations/OpTensorSyncDevice.hpp"
#include "kompute/operations/OpTensorSyncLocal.hpp"

// Will be build by CMake and placed inside the build directory
#include "kompute/shaders/generated/ShaderLogisticRegression.hpp"
#include "kompute/shaders/generated/ShaderOpMult.hpp"

#include "kompute/Version.hpp"
