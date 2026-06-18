/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <type_traits>

#include "paddle/common/ddim.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace funcs {

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

template <typename T = int64_t>
static inline T SizeToAxis(const int axis, DDim dims) {
  int64_t size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  if constexpr (std::is_same<T, int>::value ||
                std::is_same<T, int32_t>::value) {
    PADDLE_ENFORCE_LE_INT_MAX(size, "size to axis");
  }
  return static_cast<T>(size);
}

template <typename T = int64_t>
static inline T SizeFromAxis(const int axis, DDim dims) {
  int64_t size = 1;
  for (int i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  if constexpr (std::is_same<T, int>::value ||
                std::is_same<T, int32_t>::value) {
    PADDLE_ENFORCE_LE_INT_MAX(size, "size from axis");
  }
  return static_cast<T>(size);
}

template <typename T = int64_t>
static inline T SizeOutAxis(const int axis, DDim dims) {
  int64_t size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  if constexpr (std::is_same<T, int>::value ||
                std::is_same<T, int32_t>::value) {
    PADDLE_ENFORCE_LE_INT_MAX(size, "size out axis");
  }
  return static_cast<T>(size);
}

}  // namespace funcs
}  // namespace phi
