#pragma once

#include "bits_of_matcha/nn/Layer.h"


namespace matcha::nn {

struct Affine {
  unsigned units = 0;
  bool useBias = true;

  tensor operator()(const tensor& a);
  std::shared_ptr<Layer> internal_{init()};
  Layer* init();
};

}
