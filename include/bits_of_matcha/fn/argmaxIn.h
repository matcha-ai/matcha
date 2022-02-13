#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor argmaxIn(const Tensor& a);
}
}


namespace matcha {
namespace engine {
namespace fn {


class ArgmaxIn : public Fn {
  public:
    ArgmaxIn(Tensor* a);
    ArgmaxIn(const matcha::Tensor& a);
};


}
}
}
