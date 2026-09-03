// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/std_var_kernel.h"

#include <algorithm>

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

// torch's CPU var computes a one-pass Welford accumulation in `acc_t`
// (double for float/double on CPU, see at::acc_type) over each output
// element's reduce dims, iterating the reduce dims in row-major (C) order,
// and finally projects var = m2 / (nf - correction). This reproduces
// bit-for-bit the result of torch.var(x, dim=..., correction=1) on CPU.
template <typename T, typename Context>
void VarKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& axis,
               bool keepdim,
               bool unbiased,
               double correction,
               DenseTensor* out) {
  if (x.numel() == 0) {
    Full<T, Context>(dev_ctx, out->dims(), static_cast<T>(NAN), out);
    return;
  }

  const int64_t ndim = x.dims().size();

  // Normalize and deduplicate the reduce axes.
  std::vector<int64_t> axes;
  if (axis.empty() || static_cast<int64_t>(axis.size()) == ndim) {
    for (int64_t i = 0; i < ndim; ++i) axes.push_back(i);
  } else {
    for (int64_t a : axis) axes.push_back(a < 0 ? a + ndim : a);
  }
  std::sort(axes.begin(), axes.end());
  axes.erase(std::unique(axes.begin(), axes.end()), axes.end());

  const T* x_data = x.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);
  const auto& x_dims = x.dims();
  const auto& x_strides = x.strides();

  // Partition dims into keep dims and reduce dims (both ascending).
  std::vector<int64_t> keep_dims, reduce_dims;
  std::vector<bool> is_reduce(ndim, false);
  for (int64_t a : axes) is_reduce[a] = true;
  for (int64_t i = 0; i < ndim; ++i) {
    if (is_reduce[i]) {
      reduce_dims.push_back(i);
    } else {
      keep_dims.push_back(i);
    }
  }

  std::vector<int64_t> reduce_size, reduce_stride;
  int64_t reduce_numel = 1;
  for (int64_t d : reduce_dims) {
    reduce_size.push_back(x_dims[d]);
    reduce_stride.push_back(x_strides[d]);
    reduce_numel *= x_dims[d];
  }

  const int64_t out_numel = out->numel();
  const int kd = static_cast<int>(keep_dims.size());
  const int rd = static_cast<int>(reduce_dims.size());
  std::vector<int64_t> keep_size, keep_stride;
  for (int64_t d : keep_dims) {
    keep_size.push_back(x_dims[d]);
    keep_stride.push_back(x_strides[d]);
  }

  std::vector<int64_t> coord(kd, 0);
  for (int64_t oi = 0; oi < out_numel; ++oi) {
    int64_t base = 0;
    for (int j = 0; j < kd; ++j) {
      base += coord[j] * keep_stride[j];
    }

    double mean = 0;
    double m2 = 0;
    int64_t cnt = 0;

    // One-pass Welford over the reduce dims in row-major order.
    std::vector<int64_t> rcoord(rd, 0);
    for (int64_t r = 0; r < reduce_numel; ++r) {
      int64_t off = base;
      for (int j = 0; j < rd; ++j) {
        off += rcoord[j] * reduce_stride[j];
      }
      const double data = static_cast<double>(x_data[off]);
      cnt += 1;
      const double new_nf = static_cast<double>(cnt);
      const double delta = data - mean;
      const double new_mean = mean + delta / new_nf;
      const double new_delta = data - new_mean;
      m2 += delta * new_delta;
      mean = new_mean;

      for (int j = rd - 1; j >= 0; --j) {
        if (++rcoord[j] < reduce_size[j]) break;
        rcoord[j] = 0;
      }
    }

    const double nf = static_cast<double>(cnt);
    const double divisor = nf > correction ? nf - correction : 0;
    out_data[oi] = static_cast<T>(m2 / divisor);

    for (int j = kd - 1; j >= 0; --j) {
      if (++coord[j] < keep_size[j]) break;
      coord[j] = 0;
    }
  }
}

template <typename T, typename Context>
void StdKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& axis,
               bool keepdim,
               bool unbiased,
               double correction,
               DenseTensor* out) {
  if (x.numel() == 0) {
    Full<T, Context>(dev_ctx, out->dims(), static_cast<T>(NAN), out);
    return;
  }
  VarKernel<T, Context>(dev_ctx, x, axis, keepdim, unbiased, correction, out);
  SqrtKernel<T, Context>(dev_ctx, *out, out);
}

}  // namespace phi
PD_REGISTER_KERNEL(var, CPU, ALL_LAYOUT, phi::VarKernel, float, double) {}
PD_REGISTER_KERNEL(std, CPU, ALL_LAYOUT, phi::StdKernel, float, double) {}
