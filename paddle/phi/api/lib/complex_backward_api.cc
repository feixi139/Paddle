// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/api/backward/complex_backward_api.h"

#include <memory>
#include <tuple>

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/core/kernel_registry.h"

COMMON_DECLARE_bool(benchmark);

namespace paddle {
namespace experimental {

PADDLE_API std::tuple<paddle::Tensor, paddle::Tensor> complex_mul_real_grad(
    const paddle::Tensor& x,
    const paddle::Tensor& y,
    const paddle::Tensor& dout,
    int axis,
    bool need_dx,
    bool need_dy) {
  // ---- Output tensors ----
  paddle::Tensor dx;
  paddle::Tensor dy;

  paddle::Tensor* dx_ptr = need_dx ? &dx : nullptr;
  paddle::Tensor* dy_ptr = need_dy ? &dy : nullptr;

  // ---- Kernel Key Construction ----
  // x is the complex tensor; use its dtype to select the kernel.
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  {
    // Parse backend / layout from all three inputs.
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, dout);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    // Force dtype to be that of the complex tensor x so the correct
    // ComplexMulRealGradKernel specialisation is selected.
    kernel_data_type = ParseDataType(x);
  }

  VLOG(4) << "complex_mul_real_grad API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";

  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "complex_mul_real_grad",
      {kernel_backend, kernel_layout, kernel_data_type},
      true);
  const auto& kernel = kernel_result.kernel;
  VLOG(4) << "complex_mul_real_grad kernel: " << kernel;

  Backend actual_kernel_backend =
      kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
  auto* dev_ctx = GetDeviceContextByBackend(actual_kernel_backend);

  // ---- Prepare dense inputs ----
  auto input_x = PrepareData(
      x,
      GetKernelInputArgDef(kernel.InputAt(0), actual_kernel_backend),
      {},
      kernel_result.is_stride_kernel);
  auto input_y = PrepareData(
      y,
      GetKernelInputArgDef(kernel.InputAt(1), actual_kernel_backend),
      {},
      kernel_result.is_stride_kernel);
  auto input_dout = PrepareData(
      dout,
      GetKernelInputArgDef(kernel.InputAt(2), actual_kernel_backend),
      {},
      kernel_result.is_stride_kernel);

  // ---- Prepare dense outputs ----
  auto kernel_out_0 = SetKernelOutput(dx_ptr);
  auto kernel_out_1 = SetKernelOutput(dy_ptr);

  // ---- Kernel call ----
  using kernel_signature = void (*)(const phi::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    const phi::DenseTensor&,
                                    int,
                                    phi::DenseTensor*,
                                    phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

  phi::RecordEvent* kernel_record_event = nullptr;
  if (phi::RecordEvent::IsEnabled()) {
    kernel_record_event =
        new phi::RecordEvent("complex_mul_real_grad kernel launch",
                             phi::TracerEventType::DygraphKernelLaunch,
                             1);
  }
  (*kernel_fn)(*dev_ctx,
               *input_x,
               *input_y,
               *input_dout,
               axis,
               kernel_out_0,
               kernel_out_1);
  if (FLAGS_benchmark) {
    dev_ctx->Wait();
    std::cout << "complex_mul_real_grad kernel run finish." << std::endl;
  }
  if (kernel_record_event != nullptr) {
    delete kernel_record_event;
  }

  if (kernel_result.has_fallback_cpu) {
    if (kernel_out_0)
      TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    if (kernel_out_1)
      TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return std::make_tuple(dx, dy);
}

}  // namespace experimental
}  // namespace paddle
