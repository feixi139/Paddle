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

#include "paddle/phi/kernels/funcs/math/unpooling.h"

#include "paddle/common/enforce.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

namespace phi {
namespace math {
template <typename T>
__global__ void KernelUnpool2dMax(const int64_t nthreads,
                                  const T* input_data,
                                  const int* indices_data,
                                  const int input_height,
                                  const int input_width,
                                  const int channels,
                                  T* output_data,
                                  const int output_height,
                                  const int output_width) {
  CUDA_KERNEL_LOOP_TYPE(linearIndex, nthreads, int64_t) {
    int64_t c = (linearIndex / input_width / input_height) % channels;
    int64_t n = linearIndex / input_width / input_height / channels;
    output_data += (n * channels + c) * output_height * output_width;
    int maxind = indices_data[linearIndex];
    output_data[maxind] = input_data[linearIndex];
  }
}

template <typename T>
__global__ void KernelUnpool2dMaxGrad(const int64_t nthreads,
                                      const T* input_data,
                                      const int* indices_data,
                                      const int input_height,
                                      const int input_width,
                                      const int channels,
                                      const T* output_data,
                                      const T* output_grad,
                                      const int output_height,
                                      const int output_width,
                                      T* input_grad) {
  CUDA_KERNEL_LOOP_TYPE(linearIndex, nthreads, int64_t) {
    int64_t c = (linearIndex / input_width / input_height) % channels;
    int64_t n = linearIndex / input_width / input_height / channels;
    output_grad += (n * channels + c) * output_height * output_width;
    int maxind = indices_data[linearIndex];
    input_grad[linearIndex] = output_grad[maxind];
  }
}
/*
 * All tensors are in NCHW format.
 */

template <typename T>
__global__ void KernelUnpool3dMax(const int64_t nthreads,
                                  const T* input_data,
                                  const int* indices_data,
                                  const int input_depth,
                                  const int input_height,
                                  const int input_width,
                                  const int channels,
                                  T* output_data,
                                  const int output_depth,
                                  const int output_height,
                                  const int output_width) {
  CUDA_KERNEL_LOOP_TYPE(linearIndex, nthreads, int64_t) {
    int64_t c =
        (linearIndex / input_depth / input_width / input_height) % channels;
    int64_t n =
        linearIndex / input_depth / input_width / input_height / channels;
    output_data +=
        (n * channels + c) * output_depth * output_height * output_width;
    int maxind = indices_data[linearIndex];
    output_data[maxind] = input_data[linearIndex];
  }
}

template <typename T>
__global__ void KernelUnpool3dMaxGrad(const int64_t nthreads,
                                      const T* input_data,
                                      const int* indices_data,
                                      const int input_depth,
                                      const int input_height,
                                      const int input_width,
                                      const int channels,
                                      const T* output_data,
                                      const T* output_grad,
                                      const int output_depth,
                                      const int output_height,
                                      const int output_width,
                                      T* input_grad) {
  CUDA_KERNEL_LOOP_TYPE(linearIndex, nthreads, int64_t) {
    int64_t c =
        (linearIndex / input_depth / input_width / input_height) % channels;
    int64_t n =
        linearIndex / input_depth / input_width / input_height / channels;
    output_grad +=
        (n * channels + c) * output_depth * output_height * output_width;
    int maxind = indices_data[linearIndex];
    input_grad[linearIndex] = output_grad[maxind];
  }
}
/*
 * All tensors are in NCDHW format.
 */

template <typename T>
class Unpool2dMaxFunctor<GPUContext, T> {
 public:
  void operator()(const GPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& indices,
                  DenseTensor* output) {
    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t batch_size = input.dims()[0];

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t input_height_64 = input.dims()[2];
    PADDLE_ENFORCE_LE_INT_MAX(input_height_64,
                              "KernelUnpool2dMax input_height");
    int input_height = static_cast<int>(input_height_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t input_width_64 = input.dims()[3];
    PADDLE_ENFORCE_LE_INT_MAX(input_width_64, "KernelUnpool2dMax input_width");
    int input_width = static_cast<int>(input_width_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_channels_64 = output->dims()[1];
    PADDLE_ENFORCE_LE_INT_MAX(output_channels_64, "KernelUnpool2dMax channels");
    int output_channels = static_cast<int>(output_channels_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_height_64 = output->dims()[2];
    PADDLE_ENFORCE_LE_INT_MAX(output_height_64,
                              "KernelUnpool2dMax output_height");
    int output_height = static_cast<int>(output_height_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_width_64 = output->dims()[3];
    PADDLE_ENFORCE_LE_INT_MAX(output_width_64,
                              "KernelUnpool2dMax output_width");
    int output_width = static_cast<int>(output_width_64);

    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    T* output_data = context.template Alloc<T>(output);
    uint32_t threads = 1024;
    int64_t max_grid = context.GetCUDAMaxGridDimSize()[0];
    int64_t grid64 =
        std::min((input.numel() + static_cast<int64_t>(threads) - 1) /
                     static_cast<int64_t>(threads),
                 max_grid);
    PADDLE_ENFORCE_LE_UINT32_MAX(grid64, "KernelUnpool2dMax grid.x");
    uint32_t grid = static_cast<uint32_t>(grid64);
    KernelUnpool2dMax<T>
        <<<grid, threads, 0, context.stream()>>>(input.numel(),
                                                 input_data,
                                                 indices_data,
                                                 input_height,
                                                 input_width,
                                                 output_channels,
                                                 output_data,
                                                 output_height,
                                                 output_width);
  }
};
/*
 * All tensors are in NCHW format.
 */
template <typename T>
class Unpool2dMaxGradFunctor<GPUContext, T> {
 public:
  void operator()(const GPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& indices,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  DenseTensor* input_grad) {
    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t batch_size = input.dims()[0];

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t input_height_64 = input.dims()[2];
    PADDLE_ENFORCE_LE_INT_MAX(input_height_64,
                              "KernelUnpool2dMaxGrad input_height");
    int input_height = static_cast<int>(input_height_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t input_width_64 = input.dims()[3];
    PADDLE_ENFORCE_LE_INT_MAX(input_width_64,
                              "KernelUnpool2dMaxGrad input_width");
    int input_width = static_cast<int>(input_width_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_channels_64 = output.dims()[1];
    PADDLE_ENFORCE_LE_INT_MAX(output_channels_64,
                              "KernelUnpool2dMaxGrad channels");
    int output_channels = static_cast<int>(output_channels_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_height_64 = output.dims()[2];
    PADDLE_ENFORCE_LE_INT_MAX(output_height_64,
                              "KernelUnpool2dMaxGrad output_height");
    int output_height = static_cast<int>(output_height_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_width_64 = output.dims()[3];
    PADDLE_ENFORCE_LE_INT_MAX(output_width_64,
                              "KernelUnpool2dMaxGrad output_width");
    int output_width = static_cast<int>(output_width_64);

    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);
    uint32_t threads = 1024;
    int64_t max_grid = context.GetCUDAMaxGridDimSize()[0];
    int64_t grid64 =
        std::min((input.numel() + static_cast<int64_t>(threads) - 1) /
                     static_cast<int64_t>(threads),
                 max_grid);
    PADDLE_ENFORCE_LE_UINT32_MAX(grid64, "KernelUnpool2dMaxGrad grid.x");
    uint32_t grid = static_cast<uint32_t>(grid64);
    KernelUnpool2dMaxGrad<T>
        <<<grid, threads, 0, context.stream()>>>(input.numel(),
                                                 input_data,
                                                 indices_data,
                                                 input_height,
                                                 input_width,
                                                 output_channels,
                                                 output_data,
                                                 output_grad_data,
                                                 output_height,
                                                 output_width,
                                                 input_grad_data);
  }
};

template <typename T>
class Unpool3dMaxFunctor<GPUContext, T> {
 public:
  void operator()(const GPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& indices,
                  DenseTensor* output) {
    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t batch_size = input.dims()[0];

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t input_depth_64 = input.dims()[2];
    PADDLE_ENFORCE_LE_INT_MAX(input_depth_64, "KernelUnpool3dMax input_depth");
    int input_depth = static_cast<int>(input_depth_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t input_height_64 = input.dims()[3];
    PADDLE_ENFORCE_LE_INT_MAX(input_height_64,
                              "KernelUnpool3dMax input_height");
    int input_height = static_cast<int>(input_height_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t input_width_64 = input.dims()[4];
    PADDLE_ENFORCE_LE_INT_MAX(input_width_64, "KernelUnpool3dMax input_width");
    int input_width = static_cast<int>(input_width_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_channels_64 = output->dims()[1];
    PADDLE_ENFORCE_LE_INT_MAX(output_channels_64, "KernelUnpool3dMax channels");
    int output_channels = static_cast<int>(output_channels_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_depth_64 = output->dims()[2];
    PADDLE_ENFORCE_LE_INT_MAX(output_depth_64,
                              "KernelUnpool3dMax output_depth");
    int output_depth = static_cast<int>(output_depth_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_height_64 = output->dims()[3];
    PADDLE_ENFORCE_LE_INT_MAX(output_height_64,
                              "KernelUnpool3dMax output_height");
    int output_height = static_cast<int>(output_height_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_width_64 = output->dims()[4];
    PADDLE_ENFORCE_LE_INT_MAX(output_width_64,
                              "KernelUnpool3dMax output_width");
    int output_width = static_cast<int>(output_width_64);

    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    T* output_data = context.template Alloc<T>(output);
    uint32_t threads = 1024;
    int64_t max_grid = context.GetCUDAMaxGridDimSize()[0];
    int64_t grid64 =
        std::min((input.numel() + static_cast<int64_t>(threads) - 1) /
                     static_cast<int64_t>(threads),
                 max_grid);
    PADDLE_ENFORCE_LE_UINT32_MAX(grid64, "KernelUnpool3dMax grid.x");
    uint32_t grid = static_cast<uint32_t>(grid64);
    KernelUnpool3dMax<T>
        <<<grid, threads, 0, context.stream()>>>(input.numel(),
                                                 input_data,
                                                 indices_data,
                                                 input_depth,
                                                 input_height,
                                                 input_width,
                                                 output_channels,
                                                 output_data,
                                                 output_depth,
                                                 output_height,
                                                 output_width);
  }
};
/*
 * All tensors are in NCDHW format.
 */
template <typename T>
class Unpool3dMaxGradFunctor<GPUContext, T> {
 public:
  void operator()(const GPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& indices,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  DenseTensor* input_grad) {
    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t batch_size = input.dims()[0];

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t input_depth_64 = input.dims()[2];
    PADDLE_ENFORCE_LE_INT_MAX(input_depth_64,
                              "KernelUnpool3dMaxGrad input_depth");
    int input_depth = static_cast<int>(input_depth_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t input_height_64 = input.dims()[3];
    PADDLE_ENFORCE_LE_INT_MAX(input_height_64,
                              "KernelUnpool3dMaxGrad input_height");
    int input_height = static_cast<int>(input_height_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t input_width_64 = input.dims()[4];
    PADDLE_ENFORCE_LE_INT_MAX(input_width_64,
                              "KernelUnpool3dMaxGrad input_width");
    int input_width = static_cast<int>(input_width_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_channels_64 = output.dims()[1];
    PADDLE_ENFORCE_LE_INT_MAX(output_channels_64,
                              "KernelUnpool3dMaxGrad channels");
    int output_channels = static_cast<int>(output_channels_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_depth_64 = output.dims()[2];
    PADDLE_ENFORCE_LE_INT_MAX(output_depth_64,
                              "KernelUnpool3dMaxGrad output_depth");
    int output_depth = static_cast<int>(output_depth_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_height_64 = output.dims()[3];
    PADDLE_ENFORCE_LE_INT_MAX(output_height_64,
                              "KernelUnpool3dMaxGrad output_height");
    int output_height = static_cast<int>(output_height_64);

    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t output_width_64 = output.dims()[4];
    PADDLE_ENFORCE_LE_INT_MAX(output_width_64,
                              "KernelUnpool3dMaxGrad output_width");
    int output_width = static_cast<int>(output_width_64);

    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);
    uint32_t threads = 1024;
    int64_t max_grid = context.GetCUDAMaxGridDimSize()[0];
    int64_t grid64 =
        std::min((input.numel() + static_cast<int64_t>(threads) - 1) /
                     static_cast<int64_t>(threads),
                 max_grid);
    PADDLE_ENFORCE_LE_UINT32_MAX(grid64, "KernelUnpool3dMaxGrad grid.x");
    uint32_t grid = static_cast<uint32_t>(grid64);
    KernelUnpool3dMaxGrad<T>
        <<<grid, threads, 0, context.stream()>>>(input.numel(),
                                                 input_data,
                                                 indices_data,
                                                 input_depth,
                                                 input_height,
                                                 input_width,
                                                 output_channels,
                                                 output_data,
                                                 output_grad_data,
                                                 output_depth,
                                                 output_height,
                                                 output_width,
                                                 input_grad_data);
  }
};

template class Unpool2dMaxGradFunctor<GPUContext, float>;
template class Unpool2dMaxGradFunctor<GPUContext, double>;
template class Unpool2dMaxFunctor<GPUContext, float>;
template class Unpool2dMaxFunctor<GPUContext, double>;
template class Unpool3dMaxGradFunctor<GPUContext, float>;
template class Unpool3dMaxGradFunctor<GPUContext, double>;
template class Unpool3dMaxFunctor<GPUContext, float>;
template class Unpool3dMaxFunctor<GPUContext, double>;
}  // namespace math
}  // namespace phi
