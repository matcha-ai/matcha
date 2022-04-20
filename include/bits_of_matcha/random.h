#pragma once

#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/macros/generator.h"
#include "bits_of_matcha/tensor.h"

#include <functional>


namespace matcha {
class tensor;
using Generator = std::function<tensor (const Shape&)>;
}


namespace matcha::random {


struct Uniform {
  float a = 0;
  float b = 1;
  int seed = 42;

  Generator init();
  MATCHA_GENERATOR_TAIL()
};

struct Normal {
  float m = 0;
  float sd = 1;
  int seed = 42;

  Generator init();
  MATCHA_GENERATOR_TAIL()
};

}

namespace matcha {

extern random::Uniform uniform;
extern random::Normal normal;

}