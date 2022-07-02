#pragma once

#include "bits_of_matcha/nn/Layer.h"


namespace matcha::nn {

struct BatchNorm {
  tensor operator()(const tensor& batch);
  std::shared_ptr<Layer> internal_{init()};
  Layer* init();
};

}
