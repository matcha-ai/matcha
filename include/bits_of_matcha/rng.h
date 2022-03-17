#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/Generator.h"


namespace matcha::rng {

struct Uniform {
  const float a = 0;
  const float b = 0;
  const uint64_t seed = 0;

private:
  Generator init() const;
  MA_GENERATOR_TAIL();
};

struct Normal {
  const float m = 0;
  const float sd = 1;
  const uint64_t seed = 0;

private:
  Generator init() const;
  MA_GENERATOR_TAIL();
};

using Gaussian = Normal;




}