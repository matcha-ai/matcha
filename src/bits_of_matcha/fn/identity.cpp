#include "bits_of_matcha/fn/identity.h"


namespace matcha {
namespace fn {

Tensor identity(const Tensor& a) {
  Tensor t(a.dtype(), a.shape());
  t.subst(a);
  return t;
}

}
}
