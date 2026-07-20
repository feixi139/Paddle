// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/dot_kernel.h"

#include <numeric>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
namespace phi {

template <typename T>
struct DotProductFunctor {
  DotProductFunctor(const T* x, const T* y, T* out) : x_(x), y_(y), out_(out) {}

  HOSTDEVICE void operator()(size_t i) const { out_[i] = x_[i] * y_[i]; }

  const T* x_;
  const T* y_;
  T* out_;
};

template <typename T, typename Context>
void DotKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    // x[2, 1], y[2, 0], out[2]
    Full<T, Context>(dev_ctx, out->dims(), 0, out);
    return;
  }
  if (out->numel() <= 0) {
    return;
  }
  dev_ctx.template Alloc<T>(out);
  DenseTensor product;
  product.Resize(x.dims());
  dev_ctx.template Alloc<T>(&product);
  funcs::ForRange<Context> for_range(dev_ctx, x.numel());
  for_range(DotProductFunctor<T>(x.data<T>(), y.data<T>(), product.data<T>()));

  if (out->dims().size() == 0) {
    if (x.dims().size() == 0) {
      Copy(dev_ctx, product, dev_ctx.GetPlace(), false, out);
      return;
    }
    std::vector<int64_t> reduce_dims(x.dims().size());
    std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
    SumKernel<T, Context>(
        dev_ctx, product, reduce_dims, out->dtype(), false, out);
  } else {
    std::vector<int64_t> reduce_dims{static_cast<int64_t>(x.dims().size() - 1)};
    SumKernel<T, Context>(
        dev_ctx, product, reduce_dims, out->dtype(), false, out);
  }
}
}  // namespace phi

using complex64 = phi::complex64;
using complex128 = phi::complex128;

PD_REGISTER_KERNEL(dot,
                   GPU,
                   ALL_LAYOUT,
                   phi::DotKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128,
                   phi::float16,
                   phi::bfloat16) {}
