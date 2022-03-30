#pragma once

#include "bits_of_matcha/data/Dataset.h"

#include <functional>


namespace matcha::nn {

using Solver = std::function<void (const UnaryFn& forward, const std::vector<tensor*>& params, const Dataset& dataset)>;

}