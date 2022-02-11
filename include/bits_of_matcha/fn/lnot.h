#pragma once

#include "bits_of_matcha/engine/fn.h"
#include "bits_of_matcha/engine/nodeloader.h"


namespace matcha {
class Tensor;

namespace fn {
  Tensor lnot(const Tensor& a);
}
}

matcha::Tensor operator!(const matcha::Tensor& a);

namespace matcha {
namespace engine {

class Tensor;

namespace fn {


class Lnot : public Fn {
  public:
    Lnot(Tensor* a);
    Lnot(const matcha::Tensor& a);
};


}
}
}
