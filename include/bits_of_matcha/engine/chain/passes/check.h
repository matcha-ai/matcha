#pragma once

#include "bits_of_matcha/engine/chain/Chain.h"

#include <iostream>

namespace matcha::engine {

void check(const Chain& chain, std::ostream& os = std::cerr);

}