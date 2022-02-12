#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor sum(const Tensor& a);
}
}


namespace matcha {
namespace engine {
namespace fn {


class Sum : public Fn {
  public:
    Sum(Tensor* a);
    Sum(const matcha::Tensor& a);
};


}
}
}
