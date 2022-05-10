#pragma once

#include "bits_of_matcha/nn/Layer.h"


namespace matcha::nn {

struct Flatten {
  tensor operator()(const tensor& a);
};

}