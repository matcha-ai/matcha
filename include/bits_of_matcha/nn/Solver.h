#pragma once

#include "bits_of_matcha/Flow.h"
#include "bits_of_matcha/data/Dataset.h"

#include <functional>


namespace matcha::nn {

using Solver = std::function<void (Flow& flow, const Dataset& dataset)>;

}