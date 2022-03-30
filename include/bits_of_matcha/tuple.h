#pragma once

#include "bits_of_matcha/tensor.h"

#include <vector>
#include <tuple>
#include <array>


namespace matcha {

template <unsigned Size>
using tuple = std::array<tensor, Size>;

}