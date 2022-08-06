#pragma once

#include "bits_of_matcha/nn/Optimizer.h"


namespace matcha::nn {

struct Sgd {
  tensor lr = 1e-3;

  void operator()(tensor& target, const tensor& grad) const {
    constexpr bool debug = false;
    if constexpr (debug) {
      std::cerr << "target:" << std::endl;
      std::cerr << target << std::endl;
      std::cerr << std::endl << std::endl << std::endl;
      std::cerr << "grad:" << std::endl;
      std::cerr << grad << std::endl;
      std::cerr << std::endl << std::endl << std::endl;
      std::cerr << std::endl << std::endl << std::endl;
    }
    target -= lr * grad;
  }
};

}