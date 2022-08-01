#pragma once

#include "bits_of_matcha/engine/op/Op.h"

#include <random>


namespace matcha::engine::ops {

struct Normal : Op {
  Normal(Tensor* m, Tensor* sd, const Shape& shape, size_t seed);
  static Reflection<Normal> reflection;

  void run() override;

private:
  std::mt19937 gen_;
  std::normal_distribution<float> dis_;

};

}