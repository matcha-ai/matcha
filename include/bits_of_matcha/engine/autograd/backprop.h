#pragma once

#include "bits_of_matcha/engine/lambda/Lambda.h"

namespace matcha::engine {

void backprop(Lambda& lambda, const std::vector<Tensor*>& wrt);

}