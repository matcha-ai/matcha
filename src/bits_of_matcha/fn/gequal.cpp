#include "bits_of_matcha/fn/gequal.h"
#include "bits_of_matcha/fn/greater.h"
#include "bits_of_matcha/fn/equal.h"
#include "bits_of_matcha/fn/lor.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor gequal(const Tensor& a, const Tensor& b) {
  return lor(greater(a, b), equal(a, b));
}

}
}

matcha::Tensor operator>=(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::gequal(a, b);
}
