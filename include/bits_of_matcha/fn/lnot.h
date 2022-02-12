#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor lnot(const Tensor& a);
}
}


matcha::Tensor operator!(const matcha::Tensor& a);


namespace matcha {
namespace engine {
namespace fn {


class Lnot : public Fn {
  public:
    Lnot(Tensor* a);
    Lnot(const matcha::Tensor& a);
};


}
}
}
