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

#include "paddle/phi/kernels/complex_mul_real_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/elementwise_grad_base.h"
#include "paddle/phi/kernels/gpu/elementwise_grad.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void ComplexMulRealGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              int axis,
                              DenseTensor* dx,
                              DenseTensor* dy) {
  using RealT = typename phi::dtype::Real<T>;
  constexpr DataType RealDType =
      std::is_same<RealT, float>::value ? DataType::FLOAT32 : DataType::FLOAT64;

  // dx (complex grad) = dout * y, then reduce if broadcast
  if (dx != nullptr) {
    dev_ctx.template Alloc<T>(dx);
    DenseTensor tmp_dx;
    DenseTensor* target = dx;
    if (dx->dims() != dout.dims()) {
      tmp_dx.Resize(dout.dims());
      dev_ctx.template Alloc<T>(&tmp_dx);
      target = &tmp_dx;
    }
    std::vector<const DenseTensor*> ins = {&dout, &y};
    std::vector<DenseTensor*> outs = {target};
    funcs::BroadcastKernel<T>(
        dev_ctx, ins, &outs, funcs::ComplexMulRealFunctor<T>(), axis);
    if (dx->dims() != dout.dims()) {
      ReduceWrapper<T>(dev_ctx, axis, &tmp_dx, dx);
    }
  }

  // dy (real grad) = real(dout * conj(x)), then reduce if broadcast
  if (dy != nullptr) {
    // Compute dout * conj(x) -> complex tmp
    DenseTensor tmp;
    tmp.Resize(dout.dims());
    dev_ctx.template Alloc<T>(&tmp);
    {
      std::vector<const DenseTensor*> ins = {&dout, &x};
      std::vector<DenseTensor*> outs = {&tmp};
      funcs::BroadcastKernel<T>(
          dev_ctx, ins, &outs, funcs::MultiplyGradFunctor<T>(), axis);
    }

    dy->Resize(y.dims());
    dev_ctx.Alloc(dy, RealDType);

    if (y.dims() == dout.dims()) {
      RealKernel<T, GPUContext>(dev_ctx, tmp, dy);
    } else {
      // Take real part as stride=2 float view, then reduce
      DDim strides = tmp.strides();
      for (int i = 0; i < strides.size(); ++i) {
        strides[i] *= 2;
      }
      DenseTensorMeta real_meta(RealDType, tmp.dims(), strides);
      real_meta.offset = tmp.offset();
      DenseTensor real_view(tmp.Holder(), real_meta);

      std::vector<int> reduce_dims =
          funcs::GetReduceDim(y.dims(), dout.dims(), axis);
      SumKernel<RealT, GPUContext>(
          dev_ctx, real_view, reduce_dims, RealDType, false, dy);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(complex_mul_real_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ComplexMulRealGradKernel,
                   phi::complex64,
                   phi::complex128) {}
