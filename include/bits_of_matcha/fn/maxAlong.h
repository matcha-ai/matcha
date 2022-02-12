#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor maxAlong(const Tensor& a);
}
}


namespace matcha {
namespace engine {
namespace fn {


class MaxAlong : public Fn {
  public:
    MaxAlong(Tensor* a);
    MaxAlong(const matcha::Tensor& a);
};


}
}
}
