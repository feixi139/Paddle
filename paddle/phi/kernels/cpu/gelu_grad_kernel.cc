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

#include "paddle/phi/kernels/gelu_grad_kernel.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gelu_kernel.h"

namespace phi {

template <typename T>
struct GeluGradFunctor {
  void operator()(
      const T* x, const T* dout, T* dx, int n, bool approximate) const {
    if (approximate) {
      const T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2);
      const T kBeta =
          kAlpha * static_cast<T>(GELU_CONSTANT) * static_cast<T>(3);
      for (int i = 0; i < n; ++i) {
        const T x_val = x[i];
        const T x_square = x_val * x_val;
        const T y = std::tanh(
            kAlpha *
            (static_cast<T>(GELU_CONSTANT) * x_square * x_val + x_val));
        dx[i] = static_cast<T>(0.5) * dout[i] *
                (static_cast<T>(1) + y +
                 (x_val - x_val * y * y) * (kAlpha + kBeta * x_square));
      }
    } else {
      auto erf_term = static_cast<T*>(std::malloc(n * sizeof(T)));
      std::memset(erf_term, 0, n * sizeof(T));
      auto exp_term = static_cast<T*>(std::malloc(n * sizeof(T)));
      std::memset(exp_term, 0, n * sizeof(T));

      // erf_term = 0.5 * (1 + erf(x / sqrt(2)))
      for (int i = 0; i < n; ++i) {
        erf_term[i] =
            (std::erf(x[i] * static_cast<T>(M_SQRT1_2)) + static_cast<T>(1)) *
            static_cast<T>(0.5);
      }

      // exp_term = 0.5 * sqrt(2/pi) * x * exp(-0.5 * x^2)
      for (int i = 0; i < n; ++i) {
        exp_term[i] = static_cast<T>(0.5 * M_2_SQRTPI * M_SQRT1_2) * x[i] *
                      std::exp(static_cast<T>(-0.5) * x[i] * x[i]);
        erf_term[i] += exp_term[i];
      }

      // dx = dout * erf_term
      for (int i = 0; i < n; ++i) {
        dx[i] = dout[i] * erf_term[i];
      }

      std::free(erf_term);
      std::free(exp_term);
    }
  }
};

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    bool approximate,
                    DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  if (x_grad && x_grad->numel() == 0) {
    return;
  }
  GeluGradFunctor<T> functor;
  functor(x.data<T>(),
          out_grad.data<T>(),
          x_grad->data<T>(),
          static_cast<int>(x.numel()),
          approximate);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    gelu_grad, CPU, ALL_LAYOUT, phi::GeluGradKernel, float, double) {}
