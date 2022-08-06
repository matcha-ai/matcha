#pragma once

#include "bits_of_matcha/nn/Loss.h"


namespace matcha::nn {

struct Nll {
  tensor operator()(const tensor& expected, const tensor& predicted) const;
};

}