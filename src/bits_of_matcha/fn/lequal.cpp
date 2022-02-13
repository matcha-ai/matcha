#include "bits_of_matcha/fn/lequal.h"
#include "bits_of_matcha/fn/less.h"
#include "bits_of_matcha/fn/equal.h"
#include "bits_of_matcha/fn/lor.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor lequal(const Tensor& a, const Tensor& b) {
  return lor(less(a, b), equal(a, b));
}

UnaryFn lequalWith(const Tensor& b) {
  return [=](auto& a) {
    return a <= b;
  };
}

UnaryFn lequalAgainst(const Tensor& a) {
  return [=](auto& b) {
    return a <= b;
  };
}

}
}

matcha::Tensor operator<=(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::lequal(a, b);
}
