// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/gelu_kernel.h"

#include <cmath>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
struct GeluFunctor {
  void operator()(const T* x, T* out, int n, bool approximate) const {
    if (approximate) {
      // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / \pi) * (x + 0.044715 * x^{3})))
      const T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2);
      for (int i = 0; i < n; ++i) {
        const T x_val = x[i];
        const T temp =
            std::tanh(kAlpha * (x_val + static_cast<T>(GELU_CONSTANT) * x_val *
                                            x_val * x_val));
        out[i] = x_val * static_cast<T>(0.5) * (static_cast<T>(1) + temp);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        const T erf_term = std::erf(x[i] * static_cast<T>(M_SQRT1_2));
        out[i] = x[i] * (static_cast<T>(1) + erf_term) * static_cast<T>(0.5);
      }
    }
  }
};

template <typename T, typename Context>
void GeluKernel(const Context& dev_ctx,
                const DenseTensor& x,
                bool approximate,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (out && out->numel() == 0) {
    return;
  }
  GeluFunctor<T> functor;
  functor(
      x.data<T>(), out->data<T>(), static_cast<int>(x.numel()), approximate);
}

}  // namespace phi

PD_REGISTER_KERNEL(gelu, CPU, ALL_LAYOUT, phi::GeluKernel, float, double) {}
