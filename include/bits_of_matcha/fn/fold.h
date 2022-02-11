#pragma once

#include <functional>


namespace matcha {

class Tensor;
class Stream;

namespace fn {

Tensor fold(Stream& stream, const Tensor& init, const std::function<Tensor (const Tensor&, const Tensor&)>& fn);

}
}
