#pragma once

#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/macros/generator.h"
#include "bits_of_matcha/tensor.h"

#include <functional>


namespace matcha {

struct Generator {
  Generator(const std::function<tensor (const Shape& shape)>& internal)
    : internal_(internal)
  {}

  Generator() = default;

  template <class... Dims>
  inline tensor operator()(Dims... dims) const {
    return internal_(VARARG_SHAPE(dims...));
  }

  std::function<tensor (const Shape& shape)> internal_;
};

}


namespace matcha::random {

struct Uniform {
  tensor a = (float) 0;
  tensor b = (float) 1;
  int seed = 42;

  Generator init();
  MATCHA_GENERATOR_TAIL()
};

struct Normal {
  tensor m  = (float) 0;
  tensor sd = (float) 1;
  int seed = 42;

  Generator init();
  MATCHA_GENERATOR_TAIL()
};

}

namespace matcha {

extern random::Uniform uniform;
extern random::Normal normal;

}