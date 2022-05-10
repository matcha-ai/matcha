#pragma once

#include "bits_of_matcha/engine/op/Op.h"

#include <random>


namespace matcha::engine::ops {

struct Uniform : Op {
  Uniform(Tensor* a, Tensor* b, const Shape& shape, size_t seed);
  static OpMeta<Uniform> meta;

  void run() override;

private:
  std::mt19937 gen_;

};

}
