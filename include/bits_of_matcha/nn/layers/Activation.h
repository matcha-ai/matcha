#pragma once

#include "bits_of_matcha/nn/Layer.h"

#include <string>


namespace matcha::nn {

struct Activation {
  Activation(const UnaryOp& activation);
  Activation(std::string activation);
  Activation(const char* activation);

  UnaryOp internal_;
  tensor operator()(const tensor& a);
};

}