#pragma once

#include "bits_of_matcha/nn/Layer.h"
#include "bits_of_matcha/nn/initializers.h"


namespace matcha::nn {

struct Linear {
  unsigned units = 0;
  bool useBias = true;

  tensor operator()(const tensor& batch);
  std::shared_ptr<Layer> internal_{init()};
  Layer* init();
};

}
