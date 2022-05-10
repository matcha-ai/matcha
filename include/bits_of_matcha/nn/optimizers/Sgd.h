#pragma once

#include "bits_of_matcha/nn/Optimizer.h"


namespace matcha::nn {

struct Sgd {
  BinaryOp loss;
  int epochs = 1;
  float lr = 1e-3;

  void operator()(const Dataset& ds, Flow& flow);
};

}