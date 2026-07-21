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

      auto first = static_cast<T*>(std::malloc(n * sizeof(T)));
      std::memset(first, 0, n * sizeof(T));
      auto second = static_cast<T*>(std::malloc(n * sizeof(T)));
      std::memset(second, 0, n * sizeof(T));

      // first = (0.5 * (1 + erf(x / sqrt(2))))
      funcs::CBlas<T>::AXPY(n, static_cast<T>(M_SQRT1_2), x_data, 1, first, 1);
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> first_erf_map(first, n);
      Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> x_erf_map(x_data, n);
      first_erf_map = (x_erf_map * static_cast<T>(M_SQRT1_2)).erf();
      for (int i = 0; i < n; i++) {
        first[i] = (first[i] + static_cast<T>(1)) * static_cast<T>(0.5);
      }

      // second = (0.5 * 2/sqrt(pi) * 1/sqrt(2) * x * exp(-0.5 * x^2))
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> second_map(second, n);
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> x_map(x_data, n);
      second_map = (x_map.cwiseAbs2() * static_cast<T>(-0.5)).array().exp() *
                   x_map.array();
      second_map *= static_cast<T>(0.5 * M_2_SQRTPI * M_SQRT1_2);

      // dx = dout * (first + second);
      funcs::CBlas<T>::VADD(n, first, second, first);
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> dx_map(dx_data, n);
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> dout_map(dout_data,
                                                                     n);
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> first_map(first, n);
      dx_map = dout_map.array() * first_map.array();

      std::free(first);
      std::free(second);
#else
      // gelu_grad(x) = dout * 0.5 * (1 + erf(x / sqrt(2)) + x * sqrt(2 / pi) *
      // exp(- x^2 / 2)
      if (std::is_same<T, dtype::float16>::value) {
        VLOG(4) << "cast from float16 to float before computing";
        auto casted_x = x.template cast<float>();
        auto casted_dout = dout.template cast<float>();
        auto first = static_cast<float>(0.5) *
                     (static_cast<float>(1) +
                      ((casted_x * static_cast<float>(M_SQRT1_2)).erf()));
        auto second = static_cast<float>(0.5 * M_2_SQRTPI * M_SQRT1_2) *
                      casted_x *
                      (-static_cast<float>(0.5) * casted_x.square()).exp();
        dx.device(d) = (casted_dout * (first + second)).template cast<T>();
      } else {
        auto first =
            static_cast<T>(0.5) *
            (static_cast<T>(1) + ((x * static_cast<T>(M_SQRT1_2)).erf()));

        auto second = static_cast<T>(0.5 * M_2_SQRTPI * M_SQRT1_2) * x *
                      (-static_cast<T>(0.5) * x.square()).exp();
        dx.device(d) = dout * (first + second);
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
