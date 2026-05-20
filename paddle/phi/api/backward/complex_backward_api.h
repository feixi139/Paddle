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

#pragma once

#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {

// Backward for complex * real element-wise multiply.
//   x : complex tensor
//   y : real tensor
//   dout : upstream gradient (complex, same dtype as x)
//   axis : broadcast axis (same semantics as elementwise_mul)
//   need_dx : whether dx (complex grad) is required
//   need_dy : whether dy (real grad) is required
//
// Gradient formulas:
//   dx (complex grad) = dout * y,  reduced to x.shape() if broadcast
//   dy (real   grad)  = real(dout * conj(x)), reduced to y.shape() if broadcast
PADDLE_API std::tuple<paddle::Tensor, paddle::Tensor> complex_mul_real_grad(
    const paddle::Tensor& x,
    const paddle::Tensor& y,
    const paddle::Tensor& dout,
    int axis,
    bool need_dx,
    bool need_dy);

}  // namespace experimental
}  // namespace paddle
