#pragma once

#include "bits_of_matcha/nn/Optimizer.h"

#include <map>

namespace matcha::nn {

struct Sgd {
  tensor lr = 1e-2;

  void operator()(std::map<tensor*, tensor>& gradients) const;
};

}