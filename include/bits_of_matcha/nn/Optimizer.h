#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/ops.h"

#include <functional>


namespace matcha::nn {

using Optimizer = std::function<void (tensor&, const tensor&)>;

}