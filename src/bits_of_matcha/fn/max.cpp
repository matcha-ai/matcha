#include "bits_of_matcha/fn/max.h"
#include "bits_of_matcha/fn/maxIn.h"
#include "bits_of_matcha/fn/maxBetween.h"
#include "bits_of_matcha/tensor.h"


namespace matcha {
namespace fn {

Tensor max(const Tensor& a) {
  return maxIn(a);
}

Tensor max(const Tensor& a, const Tensor& b) {
  return maxBetween(a, b);
}

}
}
