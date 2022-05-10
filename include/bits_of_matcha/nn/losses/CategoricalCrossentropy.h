#pragma once

#include "bits_of_matcha/tensor.h"


namespace matcha::nn {

struct CategoricalCrossentropy {
  tensor operator()(const tensor& gold, const tensor& pred) {
    return pred - gold;
  }
};

}