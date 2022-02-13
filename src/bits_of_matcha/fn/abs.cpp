#include "bits_of_matcha/fn/abs.h"
#include "bits_of_matcha/fn/square.h"
#include "bits_of_matcha/fn/sqrt.h"


namespace matcha {
namespace fn {

Tensor abs(const Tensor& a) {
  return sqrt(square(a));
}

}
}
