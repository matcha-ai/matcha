#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/Flow.h"
#include "bits_of_matcha/dataset/Dataset.h"

#include <functional>


namespace matcha::nn {

using Optimizer = std::function<void (const Dataset& ds, Flow& flow)>;

}