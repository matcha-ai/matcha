#pragma once

#include "bits_of_matcha/tensor.h"

#include <vector>


namespace matcha {

template <class... Tensors>
inline std::vector<tensor> VARARG_TENSORS(Tensors... tensors);

template <class... Tensors>
inline void BUILD_VARARG_TENSORS(Tensors... tensors);

static thread_local std::vector<tensor> VARARG_TENSORS_BUFFER;

template <class Tensor, class... Tensors>
inline void BUILD_VARARG_TENSORS(Tensor tensor, Tensors... tensors) {
  VARARG_TENSORS_BUFFER.push_back(tensor);
  BUILD_VARARG_TENSORS(tensors...);
}

template <>
inline void BUILD_VARARG_TENSORS() {
}

template <class... Tensors>
inline std::vector<tensor> VARARG_TENSORS(Tensors... tensors) {
  VARARG_TENSORS_BUFFER.clear();
  BUILD_VARARG_TENSORS(tensors...);
  return VARARG_TENSORS_BUFFER;
}

}