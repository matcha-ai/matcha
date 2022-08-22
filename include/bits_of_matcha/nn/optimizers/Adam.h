#pragma once

#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/nn/Optimizer.h"

#include <map>

namespace matcha::nn {

struct Adam {
  tensor lr = 1e-3;
  tensor beta1 = 0.9;
  tensor beta2 = 0.999;
  tensor epsilon = 1e-7;

  void operator()(std::map<tensor*, tensor>& gradients);

  struct Internal {
    tensor t_ = 0;
    std::map<tensor*, tensor> s_;
    std::map<tensor*, tensor> r_;
  } internal_;
};

}