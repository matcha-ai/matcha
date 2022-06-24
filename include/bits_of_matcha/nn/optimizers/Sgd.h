#pragma once

#include "bits_of_matcha/nn/Optimizer.h"


namespace matcha::nn {

struct Sgd {
  float lr = 1e-3;

  void operator()(tensor& target, const tensor& grad) {
    target -= lr * grad;
  }
};

}