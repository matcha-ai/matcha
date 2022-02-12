#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/stream.h"

#include <functional>


namespace matcha {
namespace fn {

Tensor fold(Stream& stream, const Tensor& init, std::function<Tensor (const Tensor&, const Tensor&)> fn);

}
}
