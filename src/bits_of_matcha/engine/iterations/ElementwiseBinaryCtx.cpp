#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"
#include "bits_of_matcha/error/BroadcastError.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

ElementwiseBinaryCtx::ElementwiseBinaryCtx(const Shape& a, const Shape& b)
  : dims_c(std::max(a.rank(), b.rank()))
  , strides_a(dims_c.size() + 1)
  , strides_b(dims_c.size() + 1)
  , strides_c(dims_c.size() + 1)
{
  strides_a.back() = 1;
  strides_b.back() = 1;
  strides_c.back() = 1;

  for (int i = 0; i < dims_c.size(); i++) {
    int j = (int) dims_c.size() - 1 - i;
    unsigned dim_a = i < a.rank() ? a[-1 - i] : 1;
    unsigned dim_b = i < b.rank() ? b[-1 - i] : 1;
    unsigned dim_c;

    if (dim_a == dim_b) {
      dim_c = dim_a;
    } else if (dim_a == 1) {
      dim_c = dim_b;
    } else if (dim_b == 1) {
      dim_c = dim_a;
    } else {
      throw BroadcastError(a, b, -1 - i);
    }
    dims_c[j] = dim_c;

    strides_a[j] = strides_a[j + 1] * dim_a;
    strides_b[j] = strides_b[j + 1] * dim_b;
    strides_c[j] = strides_c[j + 1] * dim_c;
  }

  for (int i = 0; i < dims_c.size(); i++) {
    int j = (int) dims_c.size() - 1 - i;
    unsigned dim_a = i < a.rank() ? a[-1 - i] : 1;
    unsigned dim_b = i < b.rank() ? b[-1 - i] : 1;

    if (dim_a == dim_b) {
    } else if (dim_a == 1) {
      strides_a[j + 1] = 0;
    } else if (dim_b == 1) {
      strides_b[j + 1] = 0;
    }
  }
}

}