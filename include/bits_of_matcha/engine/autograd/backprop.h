#pragma once

#include "bits_of_matcha/engine/chain/Chain.h"

namespace matcha::engine {

void backprop(Chain& chain, const std::vector<Tensor*>& wrt);

}