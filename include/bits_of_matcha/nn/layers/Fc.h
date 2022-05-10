#pragma once

#include "bits_of_matcha/nn/Layer.h"
#include "bits_of_matcha/nn/layers/Affine.h"
#include "bits_of_matcha/nn/layers/Activation.h"


namespace matcha::nn {

struct Fc {
  unsigned units = 0;
  Activation activation = "none";
  bool useBias = true;

  tensor operator()(const tensor& a);
  UnaryOp op_ = init();
  UnaryOp init();
};

}