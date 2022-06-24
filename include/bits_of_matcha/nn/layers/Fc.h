#pragma once

#include "bits_of_matcha/nn/Layer.h"
#include "bits_of_matcha/nn/layers/Linear.h"
#include "bits_of_matcha/nn/layers/Activation.h"


namespace matcha::nn {

struct Fc {
  unsigned units = 0;
  std::string flags = "";

  tensor operator()(const tensor& batch);
  std::shared_ptr<Layer> internal_{init()};
  Layer* init();
};

}