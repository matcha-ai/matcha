#include "bits_of_matcha/fn/argmin.h"
#include "bits_of_matcha/fn/argmax.h"
#include "bits_of_matcha/fn/subtract.h"
#include "bits_of_matcha/tensor.h"


namespace matcha {
namespace fn {

Tensor argmin(const Tensor& a) {
  return argmax(-a);
}

}
}
