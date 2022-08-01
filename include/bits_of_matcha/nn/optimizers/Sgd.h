#pragma once

#include "bits_of_matcha/nn/Optimizer.h"


namespace matcha::nn {

struct Sgd {
  tensor lr = 1e-3;

  void operator()(tensor& target, const tensor& grad) const {
    target.assign(target - lr * grad);
  }
};

}