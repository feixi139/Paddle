/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "glog/logging.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/elementwise_grad_base.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T>
void ReduceWrapper(const GPUContext &dev_ctx,
                   int axis,
                   DenseTensor *src,
                   DenseTensor *dst) {
  std::vector<int> reduce_dims =
      funcs::GetReduceDim(dst->dims(), src->dims(), axis);
  SumKernel<T, GPUContext>(
      dev_ctx, *src, reduce_dims, src->dtype(), false, dst);
}

// Detect if a complex tensor was promoted from a real type by checking
// whether its first element has zero imaginary part.
// Returns false if the tensor has no data (e.g., add/sub grad where
// inputs are saved without buffers).
template <typename T>
bool IsOriginallyReal(const GPUContext &dev_ctx, const DenseTensor &t) {
  if constexpr (std::is_same<T, phi::complex64>::value ||
                std::is_same<T, phi::complex128>::value) {
    if (t.numel() == 0) return true;
    if (!t.Holder() || t.Holder()->size() == 0) return false;
    T val;
    phi::memory_utils::Copy(phi::CPUPlace(),
                            &val,
                            t.place(),
                            t.data<T>(),
                            sizeof(T),
                            dev_ctx.stream());
    dev_ctx.Wait();
    return val.imag == 0;
  } else {
    return true;
  }
}

// Reduce a complex tensor's real part using a strided view, producing a
// real-valued gradient stored in complex format (imag=0).
// This matches PyTorch's reduction order for mixed real/complex gradients.
template <typename T>
void ReduceAsReal(const GPUContext &dev_ctx,
                  const DenseTensor &src,
                  DenseTensor *dst,
                  int axis) {
  using RealT = phi::dtype::Real<T>;
  constexpr DataType real_dt = CppTypeToDataType<RealT>::Type();
  constexpr DataType complex_dt = CppTypeToDataType<T>::Type();

  DenseTensor real_view(src.Holder(), DenseTensorMeta(real_dt, src.dims()));
  real_view.set_offset(src.offset());
  auto strides_vec = vectorize(src.strides());
  for (auto &s : strides_vec) {
    s *= 2;
  }
  real_view.set_strides(make_ddim(strides_vec));

  auto sum_result =
      Sum<RealT>(dev_ctx,
                 real_view,
                 IntArray(funcs::GetReduceDim(dst->dims(), src.dims(), axis)),
                 DataType::UNDEFINED,
                 false);
  CastKernel<RealT>(dev_ctx, sum_result, complex_dt, dst);
}

// Reduce a complex tensor using strided views for real and imag parts
// separately, then combine into a complex result. This avoids depending on
// IsOriginallyReal (which requires input data) and produces reduction order
// consistent with PyTorch for both real-promoted and genuinely-complex inputs.
template <typename T, typename RealT>
static __global__ void CombineRealImagKernel(const RealT *real_data,
                                             const RealT *imag_data,
                                             T *dst,
                                             int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = T(real_data[idx], imag_data[idx]);
  }
}

template <typename T>
static __global__ void NegateKernel(T *data, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = -data[idx];
  }
}

template <typename T>
void ReduceAsComplex(const GPUContext &dev_ctx,
                     const DenseTensor &src,
                     DenseTensor *dst,
                     int axis) {
  using RealT = phi::dtype::Real<T>;
  constexpr DataType real_dt = CppTypeToDataType<RealT>::Type();

  auto reduce_dims =
      IntArray(funcs::GetReduceDim(dst->dims(), src.dims(), axis));

  auto strides_vec = vectorize(src.strides());
  for (auto &s : strides_vec) {
    s *= 2;
  }
  auto strided_ddim = make_ddim(strides_vec);

  // Strided real view (offset=0, stride×2)
  DenseTensor real_view(src.Holder(), DenseTensorMeta(real_dt, src.dims()));
  real_view.set_offset(src.offset());
  real_view.set_strides(strided_ddim);

  // Strided imag view (offset=sizeof(RealT), stride×2)
  DenseTensor imag_view(src.Holder(), DenseTensorMeta(real_dt, src.dims()));
  imag_view.set_offset(src.offset() + sizeof(RealT));
  imag_view.set_strides(strided_ddim);

  auto real_sum =
      Sum<RealT>(dev_ctx, real_view, reduce_dims, DataType::UNDEFINED, false);
  auto imag_sum =
      Sum<RealT>(dev_ctx, imag_view, reduce_dims, DataType::UNDEFINED, false);

  // Combine real and imag sums into complex dst
  dev_ctx.template Alloc<T>(dst);
  auto numel = dst->numel();
  const RealT *real_ptr = real_sum.template data<RealT>();
  const RealT *imag_ptr = imag_sum.template data<RealT>();
  T *dst_ptr = dst->template data<T>();
  dim3 block(PREDEFINED_BLOCK_SIZE);
  dim3 grid((numel + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE);
  CombineRealImagKernel<T, RealT><<<grid, block, 0, dev_ctx.stream()>>>(
      real_ptr, imag_ptr, dst_ptr, numel);
}

template <typename T, typename Functor>
void GetGradXAndYOut(const GPUContext &dev_ctx,
                     const Place &place,
                     int axis,
                     std::vector<const DenseTensor *> ins,
                     const DenseTensor &dout,
                     DenseTensor *dx,
                     DenseTensor *dy,
                     Functor func) {
  DenseTensor tmp_dx;
  DenseTensor tmp_dy;
  dev_ctx.Alloc<T>(dx);
  dev_ctx.Alloc<T>(dy);
  std::vector<DenseTensor *> outs;
  if (dx->dims() == dout.dims() && dy->dims() == dout.dims()) {
    outs = {dx, dy};
  } else if (dx->dims() != dout.dims() && dy->dims() == dout.dims()) {
    tmp_dx.Resize(dout.dims());
    dev_ctx.Alloc<T>(&tmp_dx);
    outs = {&tmp_dx, dy};
  } else if (dx->dims() == dout.dims() && dy->dims() != dout.dims()) {
    tmp_dy.Resize(dout.dims());
    dev_ctx.Alloc<T>(&tmp_dy);
    outs = {dx, &tmp_dy};
  } else if (dx->dims() != dout.dims() && dy->dims() != dout.dims()) {
    tmp_dy.Resize(dout.dims());
    dev_ctx.Alloc<T>(&tmp_dy);
    tmp_dx.Resize(dout.dims());
    dev_ctx.Alloc<T>(&tmp_dx);
    outs = {&tmp_dx, &tmp_dy};
  }

  funcs::BroadcastKernel<T, decltype(func), 2>(dev_ctx, ins, &outs, func, axis);

  if (dx->dims() != dout.dims() && dy->dims() == dout.dims()) {
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dx, dx);
  } else if (dx->dims() == dout.dims() && dy->dims() != dout.dims()) {
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dy, dy);
  } else if (dx->dims() != dout.dims() && dy->dims() != dout.dims()) {
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dx, dx);
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dy, dy);
  }
}

template <typename T, typename Functor>
void GetGradXOrYOut(const GPUContext &dev_ctx,
                    const Place &place,
                    int axis,
                    std::vector<const DenseTensor *> ins,
                    const DenseTensor &dout,
                    DenseTensor *dxy,
                    Functor func) {
  DenseTensor tmp_dxy;
  dev_ctx.Alloc<T>(dxy);

  std::vector<DenseTensor *> outs;
  if (dxy->dims() != dout.dims()) {
    tmp_dxy.Resize(dout.dims());
    dev_ctx.Alloc<T>(&tmp_dxy);
    outs = {&tmp_dxy};
  } else {
    outs = {dxy};
  }

  funcs::BroadcastKernel<T>(dev_ctx, ins, &outs, func, axis);
  if (dxy->dims() != dout.dims()) {
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dxy, dxy);
  }
}

/*
******************************
    Add Grad
******************************
*/

template <typename T>
struct alignas(sizeof(T) * 4) Pack4 {
  T val[4];
};

template <typename T_dy, typename IndexT = int>
static __global__ void MixedPrecisionElemwiseAddGradCUDAKernel(
    const float *__restrict__ dout,
    IndexT size,
    float *__restrict__ dx,
    T_dy *__restrict__ dy) {
  IndexT tid = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x;
  IndexT stride = static_cast<IndexT>(gridDim.x) * blockDim.x;

  constexpr int vec_size = 4;
  IndexT loop = size / vec_size;
  IndexT remainder = size % vec_size;

  const float4 *__restrict__ dout_vec = reinterpret_cast<const float4 *>(dout);
  float4 *__restrict__ dx_vec = reinterpret_cast<float4 *>(dx);
  Pack4<T_dy> *__restrict__ dy_vec = reinterpret_cast<Pack4<T_dy> *>(dy);

  for (IndexT i = tid; i < loop; i += stride) {
    float4 val = __ldg(dout_vec + i);
    dx_vec[i] = val;

    Pack4<T_dy> dy_pack;
    dy_pack.val[0] = static_cast<T_dy>(val.x);
    dy_pack.val[1] = static_cast<T_dy>(val.y);
    dy_pack.val[2] = static_cast<T_dy>(val.z);
    dy_pack.val[3] = static_cast<T_dy>(val.w);
    dy_vec[i] = dy_pack;
  }

  if (remainder != 0) {
    IndexT tail_start = loop * vec_size;
    for (IndexT i = tail_start + tid; i < size; i += stride) {
      float val = __ldg(dout + i);
      dx[i] = val;
      dy[i] = static_cast<T_dy>(val);
    }
  }
}

template <typename T_dy>
void ElementwiseMixedPrecisionAddGrad(const GPUContext &dev_ctx,
                                      const DenseTensor &dout,
                                      DenseTensor *dx,
                                      DenseTensor *dy) {
  using T_dout = float;
  using T_dx = float;

  auto *dx_data = dev_ctx.template Alloc<T_dx>(dx);
  T_dy *dy_data = dev_ctx.template Alloc<T_dy>(dy);
  auto *dout_data = dout.data<T_dout>();

  if (dx_data == dout_data) {
    VLOG(7) << "Special case when dx_data is the same as dout_data, "
               "need cast dout to dy.";
    CastKernel<T_dout>(dev_ctx, dout, dy->dtype(), dy);
    return;
  }

  auto size = dout.numel();
  if (size == 0) return;

  constexpr int vec_size = 4;
  const int64_t main_size = (size / vec_size) * vec_size;
  const int block_size = PREDEFINED_BLOCK_SIZE;
  const int grid_size =
      std::min(static_cast<int>((main_size + block_size - 1) / block_size),
               (dev_ctx.GetMaxPhysicalThreadCount() / block_size));

  dim3 grid_dim(grid_size, 1, 1);
  dim3 block_dim(block_size, 1, 1);

  if (size < std::numeric_limits<int>::max()) {
    MixedPrecisionElemwiseAddGradCUDAKernel<T_dy, int>
        <<<grid_dim, block_dim, 0, dev_ctx.stream()>>>(
            dout_data, static_cast<int>(size), dx_data, dy_data);
  } else {
    MixedPrecisionElemwiseAddGradCUDAKernel<T_dy, int64_t>
        <<<grid_dim, block_dim, 0, dev_ctx.stream()>>>(
            dout_data, static_cast<int64_t>(size), dx_data, dy_data);
  }
}

template <typename T_dy>
void DefaultMixedPrecisionAddGrad(const GPUContext &dev_ctx,
                                  const DenseTensor &x,
                                  const DenseTensor &y,
                                  const DenseTensor &dout,
                                  DenseTensor *dx,
                                  DenseTensor *dy,
                                  int axis = -1) {
  using T_dout = float;
  using T_dx = float;

  auto *dout_data = dout.data<T_dout>();

  // dx
  if (dx != nullptr) {
    auto *dx_data = dev_ctx.template Alloc<T_dx>(dx);
    if (dx->dims() == dout.dims()) {
      if (dx_data != dout_data) {
        Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
      }
    } else {
      if (dx->IsSharedBufferWith(dout)) {
        dx->clear();
        dx->Resize(x.dims());
        dev_ctx.template Alloc<T_dx>(dx);
      }
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(x.dims(), dout.dims(), axis);
      SumKernel<T_dout, GPUContext>(
          dev_ctx, dout, reduce_dims, dout.dtype(), false, dx);
    }
  }

  // dy
  if (dy != nullptr) {
    auto *dy_data = dev_ctx.template Alloc<T_dy>(dy);
    if (dy->dims() == dout.dims()) {
      CastKernel<T_dout>(dev_ctx, dout, dy->dtype(), dy);
    } else {
      DenseTensor dy_fp32;
      dy_fp32.Resize(dout.dims());
      dev_ctx.template Alloc<float>(&dy_fp32);
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(y.dims(), dout.dims(), axis);
      SumKernel<float, GPUContext>(
          dev_ctx, dout, reduce_dims, dout.dtype(), false, &dy_fp32);
      CastKernel<float>(dev_ctx, dy_fp32, dy->dtype(), dy);
    }
  }
}

template <typename T, typename IndexT = int>
static __global__ void SimpleElemwiseAddGradCUDAKernel(
    const T *__restrict__ dout, IndexT size, int vec_size, T *dx, T *dy) {
  IndexT tid = static_cast<IndexT>(BLOCK_ID_X) * BLOCK_NUM_X + THREAD_ID_X;
  IndexT stride = static_cast<IndexT>(GRID_NUM_X) * BLOCK_NUM_X;
  IndexT loop = size / vec_size;
  IndexT remainder = size % vec_size;
  const float4 *dout_vec = reinterpret_cast<const float4 *>(dout);
  float4 *dx_vec = reinterpret_cast<float4 *>(dx);
  float4 *dy_vec = reinterpret_cast<float4 *>(dy);
  float4 tmp_loop;

  for (IndexT i = tid; i < loop; i += stride) {
    tmp_loop = dout_vec[i];
    dx_vec[i] = tmp_loop;
    dy_vec[i] = tmp_loop;
  }

  if (tid == loop && remainder != 0) {
    T tmp_rem;
    while (remainder) {
      IndexT idx = size - remainder;
      remainder--;
      tmp_rem = dout[idx];
      dx[idx] = tmp_rem;
      dy[idx] = tmp_rem;
    }
  }
}

template <typename T>
void DefaultElementwiseAddGrad(const GPUContext &dev_ctx,
                               const DenseTensor &x,
                               const DenseTensor &y,
                               const DenseTensor &out,
                               const DenseTensor &dout,
                               DenseTensor *dx,
                               DenseTensor *dy,
                               int axis = -1) {
  auto *dout_data = dout.data<T>();

  // dx
  if (dx != nullptr) {
    auto *dx_data = dev_ctx.template Alloc<T>(dx);
    if (dx->dims() == dout.dims()) {
      if (dx_data != dout_data) {
        Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
      }
    } else {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedBufferWith(dout)) {
        dx->clear();
        dx->Resize(x.dims());
        dev_ctx.template Alloc<T>(dx);
      }
      if constexpr (std::is_same<T, phi::complex64>::value ||
                    std::is_same<T, phi::complex128>::value) {
        ReduceAsComplex<T>(dev_ctx, dout, dx, axis);
      } else {
        std::vector<int> reduce_dims =
            funcs::GetReduceDim(x.dims(), out.dims(), axis);
        SumKernel<T, GPUContext>(
            dev_ctx, dout, reduce_dims, dout.dtype(), false, dx);
      }
    }
  }
  // dy
  if (dy != nullptr) {
    auto *dy_data = dev_ctx.template Alloc<T>(dy);
    if (dy->dims() == dout.dims()) {
      if (dy_data != dout_data) {
        Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
      }
    } else {
      if constexpr (std::is_same<T, phi::complex64>::value ||
                    std::is_same<T, phi::complex128>::value) {
        ReduceAsComplex<T>(dev_ctx, dout, dy, axis);
      } else {
        std::vector<int> reduce_dims =
            funcs::GetReduceDim(y.dims(), out.dims(), axis);
        SumKernel<T, GPUContext>(
            dev_ctx, dout, reduce_dims, dout.dtype(), false, dy);
      }
    }
  }
}

template <typename T>
void ElementwiseAddGrad(const GPUContext &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        const DenseTensor &out,
                        const DenseTensor &dout,
                        DenseTensor *dx,
                        DenseTensor *dy) {
  dev_ctx.template Alloc<T>(dx);
  dev_ctx.template Alloc<T>(dy);
  auto *dx_data = dx->data<T>();
  auto *dy_data = dy->data<T>();
  auto *dout_data = dout.data<T>();
  if (dx_data == dout_data && dy_data != dout_data) {
    VLOG(4) << "Special case when dx_data is the same as dout_data, "
               "only need copy dout to dy";
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
  } else if (dx_data != dout_data && dy_data == dout_data) {
    VLOG(4) << "Special case when dy_data is the same as dout_data, "
               "only need copy dout to dx";
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
  } else if (dx_data != dout_data && dy_data != dout_data) {
    auto size = x.numel();
    int vec_size = max(static_cast<int>(sizeof(float4) / sizeof(T)), 1);
    dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
    dim3 grid_size =
        dim3(((size + vec_size - 1) / vec_size + PREDEFINED_BLOCK_SIZE - 1) /
                 PREDEFINED_BLOCK_SIZE,
             1);
    if (size < std::numeric_limits<int>::max()) {
      SimpleElemwiseAddGradCUDAKernel<T>
          <<<grid_size, block_size, 0, dev_ctx.stream()>>>(
              dout.data<T>(),
              size,
              vec_size,
              dev_ctx.template Alloc<T>(dx),
              dev_ctx.template Alloc<T>(dy));
    } else {
      SimpleElemwiseAddGradCUDAKernel<T, int64_t>
          <<<grid_size, block_size, 0, dev_ctx.stream()>>>(
              dout.data<T>(),
              size,
              vec_size,
              dev_ctx.template Alloc<T>(dx),
              dev_ctx.template Alloc<T>(dy));
    }

  } else {
    VLOG(4) << "Special case when dy_data is the same as dout_data, "
               "and dx_data is the same as dout_data, do not need "
               "any operator";
  }
}

/*
******************************
    Sub Grad
******************************
*/

template <typename T>
static __global__ void SimpleElemwiseSubGradCUDAKernel(const T *dout,
                                                       int64_t size,
                                                       T *dx,
                                                       T *dy) {
  int64_t col = static_cast<int64_t>(BLOCK_ID_X) * BLOCK_NUM_X + THREAD_ID_X;

  while (col < size) {
    if (dx != nullptr) {
      dx[col] = dout[col];
    }
    dy[col] = -dout[col];
    col += static_cast<int64_t>(BLOCK_NUM_X) * GRID_NUM_X;
  }
}

template <typename T>
void default_elementwise_sub_grad(const GPUContext &dev_ctx,
                                  const DenseTensor &x,
                                  const DenseTensor &y,
                                  const DenseTensor &out,
                                  const DenseTensor &dout,
                                  DenseTensor *dx,
                                  DenseTensor *dy,
                                  int axis = -1) {
  auto *dout_data = dout.data<T>();

  // dx
  if (dx != nullptr) {
    auto *dx_data = dev_ctx.template Alloc<T>(dx);
    if (dx->dims() == dout.dims()) {
      if (dx_data != dout_data) {
        Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
      }
    } else {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedBufferWith(dout)) {
        dx->clear();
        dx->Resize(x.dims());
        dev_ctx.template Alloc<T>(dx);
      }
      if constexpr (std::is_same<T, phi::complex64>::value ||
                    std::is_same<T, phi::complex128>::value) {
        ReduceAsComplex<T>(dev_ctx, dout, dx, axis);
      } else {
        std::vector<int> reduce_dims =
            funcs::GetReduceDim(x.dims(), out.dims(), axis);
        SumKernel<T, GPUContext>(
            dev_ctx, dout, reduce_dims, dout.dtype(), false, dx);
      }
    }
  }
  // dy
  if (dy != nullptr) {
    auto *dy_data = dev_ctx.template Alloc<T>(dy);
    if (dy->dims() == dout.dims()) {
      if (dy_data != dout_data) {
        dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
        auto size = dy->numel();
        dim3 grid_size =
            dim3((size + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
        SimpleElemwiseSubGradCUDAKernel<T>
            <<<grid_size, block_size, 0, dev_ctx.stream()>>>(
                dout.data<T>(), size, nullptr, dev_ctx.template Alloc<T>(dy));
      }
    } else {
      if constexpr (std::is_same<T, phi::complex64>::value ||
                    std::is_same<T, phi::complex128>::value) {
        ReduceAsComplex<T>(dev_ctx, dout, dy, axis);
        // Negate dy in-place: sub grad dy = -reduce(dout)
        int64_t size = dy->numel();
        dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
        dim3 grid_size =
            dim3((size + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
        SimpleElemwiseSubGradCUDAKernel<T>
            <<<grid_size, block_size, 0, dev_ctx.stream()>>>(
                dy->data<T>(), size, nullptr, dy->data<T>());
      } else {
        std::vector<int> reduce_dims =
            funcs::GetReduceDim(y.dims(), out.dims(), axis);
        funcs::ReduceKernel<T, T, kps::AddFunctor, kps::InverseFunctor<T>>(
            dev_ctx, dout, dy, kps::InverseFunctor<T>(), reduce_dims);
      }
    }
  }
}

template <typename T>
void elementwise_sub_grad(const GPUContext &dev_ctx,
                          const DenseTensor &x,
                          const DenseTensor &y,
                          const DenseTensor &out,
                          const DenseTensor &dout,
                          DenseTensor *dx,
                          DenseTensor *dy) {
  dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
  auto size = x.numel();
  dim3 grid_size =
      dim3((size + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
  SimpleElemwiseSubGradCUDAKernel<T>
      <<<grid_size, block_size, 0, dev_ctx.stream()>>>(
          dout.data<T>(),
          size,
          dev_ctx.template Alloc<T>(dx),
          dev_ctx.template Alloc<T>(dy));
}
/*
******************************
    Div Grad
******************************
*/
template <typename T>
void ElementwiseDivGrad(const GPUContext &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        const DenseTensor &out,
                        const DenseTensor &dout,
                        DenseTensor *dx,
                        DenseTensor *dy,
                        int axis = -1) {
  const auto place = dev_ctx.GetPlace();

  if constexpr (std::is_same<T, phi::complex64>::value ||
                std::is_same<T, phi::complex128>::value) {
    bool x_is_real = IsOriginallyReal<T>(dev_ctx, x);
    bool y_is_real = IsOriginallyReal<T>(dev_ctx, y);

    bool dx_need_reduce = (dx != nullptr && dx->dims() != dout.dims());
    bool dy_need_reduce = (dy != nullptr && dy->dims() != dout.dims());

    bool need_special =
        (dx_need_reduce && x_is_real) || (dy_need_reduce && y_is_real);

    if (need_special) {
      if (dx != nullptr) {
        dev_ctx.template Alloc<T>(dx);
        // dx = dout / conj(y)
        DenseTensor y_conj = Conj<T>(dev_ctx, y);
        DenseTensor div_result;
        div_result.Resize(dout.dims());
        funcs::ElementwiseCompute<funcs::DivideFunctor<T>, T>(
            dev_ctx,
            dout,
            y_conj,
            funcs::DivideFunctor<T>(),
            &div_result,
            axis);

        if (dx_need_reduce) {
          if (x_is_real) {
            ReduceAsReal<T>(dev_ctx, div_result, dx, axis);
          } else {
            ReduceWrapper<T>(dev_ctx, axis, &div_result, dx);
          }
        } else {
          if (x_is_real) {
            using RealT = phi::dtype::Real<T>;
            constexpr DataType complex_dt = CppTypeToDataType<T>::Type();
            auto tmp_real = Real<T>(dev_ctx, div_result);
            CastKernel<RealT>(dev_ctx, tmp_real, complex_dt, dx);
          } else {
            *dx = div_result;
          }
        }
      }

      if (dy != nullptr) {
        dev_ctx.template Alloc<T>(dy);
        // dy = -dout * conj(out / y / y)
        DenseTensor y_conj = Conj<T>(dev_ctx, y);
        DenseTensor out_conj = Conj<T>(dev_ctx, out);
        // out_div_y_conj = conj(out) / conj(y)
        DenseTensor out_div_y;
        out_div_y.Resize(dout.dims());
        funcs::ElementwiseCompute<funcs::DivideFunctor<T>, T>(
            dev_ctx,
            out_conj,
            y_conj,
            funcs::DivideFunctor<T>(),
            &out_div_y,
            axis);
        // out_div_yy = out_div_y / conj(y)
        DenseTensor out_div_yy;
        out_div_yy.Resize(dout.dims());
        funcs::ElementwiseCompute<funcs::DivideFunctor<T>, T>(
            dev_ctx,
            out_div_y,
            y_conj,
            funcs::DivideFunctor<T>(),
            &out_div_yy,
            axis);
        // dy_full = dout * out_div_yy, then negate to get -dout * out_div_yy
        DenseTensor dy_full;
        dy_full.Resize(dout.dims());
        funcs::ElementwiseCompute<funcs::MultiplyFunctor<T>, T>(
            dev_ctx,
            dout,
            out_div_yy,
            funcs::MultiplyFunctor<T>(),
            &dy_full,
            axis);
        {
          T *ptr = dev_ctx.template Alloc<T>(&dy_full);
          int64_t n = dy_full.numel();
          dim3 block(PREDEFINED_BLOCK_SIZE);
          dim3 grid((n + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE);
          NegateKernel<T><<<grid, block, 0, dev_ctx.stream()>>>(ptr, n);
        }

        if (dy_need_reduce) {
          if (y_is_real) {
            ReduceAsReal<T>(dev_ctx, dy_full, dy, axis);
          } else {
            ReduceWrapper<T>(dev_ctx, axis, &dy_full, dy);
          }
        } else {
          if (y_is_real) {
            using RealT = phi::dtype::Real<T>;
            constexpr DataType complex_dt = CppTypeToDataType<T>::Type();
            auto tmp_real = Real<T>(dev_ctx, dy_full);
            CastKernel<RealT>(dev_ctx, tmp_real, complex_dt, dy);
          } else {
            *dy = dy_full;
          }
        }
      }
      return;
    }
  }

  // Default path (non-complex or no special handling needed)
  if (dx != nullptr && dy != nullptr) {
    std::vector<const DenseTensor *> ins = {&dout, &out, &y};
    GetGradXAndYOut<T>(dev_ctx,
                       place,
                       axis,
                       ins,
                       dout,
                       dx,
                       dy,
                       funcs::DivGradXYFunctor<T, T>());
  } else if (dx != nullptr && dy == nullptr) {
    std::vector<const DenseTensor *> ins = {&dout, &y};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dx, funcs::DivGradXFunctor<T>());
  } else if (dy != nullptr && dx == nullptr) {
    std::vector<const DenseTensor *> ins = {&dout, &out, &y};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dy, funcs::DivGradYFunctor<T>());
  }
}

/*
******************************
    Mul Grad
******************************
*/

template <typename T>
void ElementwiseMulGrad(const GPUContext &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        const DenseTensor &dout,
                        DenseTensor *dx,
                        DenseTensor *dy,
                        int axis) {
  if (dout.numel() == 0) {
    if (dx) {
      if (dx->numel() == 0) {
        dev_ctx.template Alloc<T>(dx);
      } else {
        Full<T, GPUContext>(dev_ctx, dx->dims(), 0, dx);
      }
    }
    if (dy) {
      if (dy->numel() == 0) {
        dev_ctx.template Alloc<T>(dy);
      } else {
        Full<T, GPUContext>(dev_ctx, dy->dims(), 0, dy);
      }
    }
    return;
  }

  // For mixed real/complex multiply, the framework promotes inputs to T
  // (complex). PyTorch takes real() BEFORE sum() for real-valued gradients,
  // while the original code did Sum<T>(complex) then the framework took
  // real() after — producing different results due to GPU reduction order.
  // We detect originally-real inputs by checking if their imaginary part is
  // zero (a real-to-complex promotion always produces zero imag parts).
  bool x_is_real = true;
  bool y_is_real = true;

  if constexpr (std::is_same<T, phi::complex64>::value ||
                std::is_same<T, phi::complex128>::value) {
    x_is_real = IsOriginallyReal<T>(dev_ctx, x);
    y_is_real = IsOriginallyReal<T>(dev_ctx, y);
  }

  auto compute_grad = [&](const DenseTensor &other,
                          DenseTensor *grad,
                          bool grad_should_be_real) {
    dev_ctx.template Alloc<T>(grad);
    DenseTensor other_conj = Conj<T>(dev_ctx, other);
    bool need_reduce = (grad->dims() != dout.dims());

    DenseTensor mul_result;
    mul_result.Resize(dout.dims());

    funcs::ElementwiseCompute<funcs::MultiplyFunctor<T>, T>(
        dev_ctx,
        dout,
        other_conj,
        funcs::MultiplyFunctor<T>(),
        &mul_result,
        axis);

    if (need_reduce) {
      if (grad_should_be_real) {
        if constexpr (std::is_same<T, phi::complex128>::value ||
                      std::is_same<T, phi::complex64>::value) {
          ReduceAsReal<T>(dev_ctx, mul_result, grad, axis);
        } else {
          ReduceWrapper<T>(dev_ctx, axis, &mul_result, grad);
        }
      } else {
        ReduceWrapper<T>(dev_ctx, axis, &mul_result, grad);
      }
    } else {
      if (grad_should_be_real) {
        if constexpr (std::is_same<T, phi::complex128>::value ||
                      std::is_same<T, phi::complex64>::value) {
          using RealT = phi::dtype::Real<T>;
          constexpr DataType complex_dt = CppTypeToDataType<T>::Type();
          auto tmp_real = Real<T>(dev_ctx, mul_result);
          CastKernel<RealT>(dev_ctx, tmp_real, complex_dt, grad);
        } else {
          *grad = mul_result;
        }
      } else {
        *grad = mul_result;
      }
    }
  };

  if (dx != nullptr && dy != nullptr) {
    compute_grad(y, dx, x_is_real);
    compute_grad(x, dy, y_is_real);
  } else if (dx != nullptr && dy == nullptr) {
    compute_grad(y, dx, x_is_real);
  } else if (dx == nullptr && dy != nullptr) {
    compute_grad(x, dy, y_is_real);
  }
}
}  // namespace phi
