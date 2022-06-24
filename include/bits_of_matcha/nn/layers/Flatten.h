#pragma once

#include "bits_of_matcha/nn/Layer.h"


namespace matcha::nn {

tensor flatten(const tensor& batch);

struct Flatten {
  tensor operator()(const tensor& batch) { return flatten(batch); }
};

}