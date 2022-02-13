#include "bits_of_matcha/fn/argmax.h"
#include "bits_of_matcha/fn/argmaxIn.h"
#include "bits_of_matcha/tensor.h"


namespace matcha {
namespace fn {

Tensor argmax(const Tensor& a) {
  return argmaxIn(a);
}

}
}
