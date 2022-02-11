#pragma once

#include "bits_of_matcha/engine/fn.h"
#include "bits_of_matcha/engine/nodeloader.h"


namespace matcha {
class Tensor;
class Stream;
class Tuple;

namespace fn {
  Tensor lor(const Tensor& a, const Tensor& b);
}
}

matcha::Tensor operator||(const matcha::Tensor& a, const matcha::Tensor& b);

namespace matcha {
namespace engine {

class Tensor;

namespace fn {


class Lor : public Fn {
  public:
    Lor(Tensor* a, Tensor* b);
    Lor(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
