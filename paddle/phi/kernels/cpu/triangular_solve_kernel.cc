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

#include "paddle/phi/kernels/triangular_solve_kernel.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"

#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T>
void SolveTriangularSystem(const T* a,
                           T* b,
                           int m,
                           int n,
                           bool upper,
                           bool transpose,
                           bool unitriangular) {
  const bool effective_upper = upper != transpose;
  auto coeff = [=](int row, int col) {
    return transpose ? a[col * m + row] : a[row * m + col];
  };
  for (int col = 0; col < n; ++col) {
    if (effective_upper) {
      for (int row = m - 1; row >= 0; --row) {
        T sum = b[row * n + col];
        for (int k = row + 1; k < m; ++k) {
          sum -= coeff(row, k) * b[k * n + col];
        }
        b[row * n + col] = unitriangular ? sum : sum / coeff(row, row);
      }
    } else {
      for (int row = 0; row < m; ++row) {
        T sum = b[row * n + col];
        for (int k = 0; k < row; ++k) {
          sum -= coeff(row, k) * b[k * n + col];
        }
        b[row * n + col] = unitriangular ? sum : sum / coeff(row, row);
      }
    }
  }
}

template <typename T, typename Context>
void TriangularSolveKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           bool upper,
                           bool transpose,
                           bool unitriangular,
                           DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  // get broadcast dim
  std::vector<int64_t> x_bst_dims_vec;
  std::vector<int64_t> y_bst_dims_vec;
  std::tie(x_bst_dims_vec, y_bst_dims_vec) =
      funcs::MatrixGetBroadcastDims(x, y);
  int x_bst_ndim = static_cast<int>(x_bst_dims_vec.size());
  int y_bst_ndim = static_cast<int>(y_bst_dims_vec.size());

  // Tensor broadcast to 'out' and temp 'x_bst'
  IntArray x_bst_dims(x_bst_dims_vec);
  DenseTensor x_bst = Empty<T, Context>(dev_ctx, x_bst_dims);
  const T* x_bst_data = x_bst.data<T>();
  ExpandKernel<T, Context>(dev_ctx, x, x_bst_dims, &x_bst);

  out->Resize(y_bst_dims_vec);
  T* out_data = dev_ctx.template Alloc<T>(out);
  IntArray y_bst_dims(y_bst_dims_vec);
  ExpandKernel<T, Context>(dev_ctx, y, y_bst_dims, out);

  // Calculate by solving each triangular system in place.
  int M = static_cast<int>(y_bst_dims_vec[y_bst_ndim - 2]);
  int N = static_cast<int>(y_bst_dims_vec[y_bst_ndim - 1]);
  int batch_size = 1;
  for (int i = 0; i < x_bst_ndim - 2; i++) {
    batch_size *= static_cast<int>(x_bst_dims_vec[i]);
  }

  for (int i = 0; i < batch_size; ++i) {
    const auto* x_matrix_ptr = x_bst_data + i * M * M;
    auto* out_matrix_ptr = out_data + i * M * N;
    SolveTriangularSystem<T>(
        x_matrix_ptr, out_matrix_ptr, M, N, upper, transpose, unitriangular);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(triangular_solve,
                   CPU,
                   ALL_LAYOUT,
                   phi::TriangularSolveKernel,
                   float,
                   double,
                   phi::complex64,
                   phi::complex128) {}
