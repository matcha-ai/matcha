#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/Tensor.h"
#include "bits_of_matcha/macros/generator.h"

#include <functional>


namespace matcha {

class Generator {
public:
  Generator(const std::function<tensor(const Shape&)> generator);

  tensor operator()(const Shape& shape) const;

  template <class... Dims>
  inline tensor operator()(Dims... dims) const {
    return operator()(VARARG_SHAPE(dims...));
  }

private:
  std::function<tensor(const Shape&)> generator_;
};

}