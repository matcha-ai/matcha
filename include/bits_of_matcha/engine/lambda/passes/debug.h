#pragma once

#include "bits_of_matcha/engine/lambda/Lambda.h"

#include <iostream>

namespace matcha::engine {

void debug(const Lambda& lambda, std::ostream& os = std::cerr);

}