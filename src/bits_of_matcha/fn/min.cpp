#include "bits_of_matcha/fn/min.h"
#include "bits_of_matcha/fn/max.h"
#include "bits_of_matcha/fn/subtract.h"


namespace matcha {
namespace fn {

Tensor min(const Tensor& a) {
  return -max(-a);
}

Tensor min(const Tensor& a, const Tensor& b) {
  return -max(-a, -b);
}

}
}
