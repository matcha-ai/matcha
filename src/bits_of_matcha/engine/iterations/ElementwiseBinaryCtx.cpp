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
  strides_a[strides_a.size() - 1] = 1;
  strides_b[strides_b.size() - 1] = 1;
  strides_c[strides_c.size() - 1] = 1;

  for (int i = 0; i < dims_c.size(); i++) {
    int j = (int) dims_c.size() - 1 - i;
    unsigned dimA = i < a.rank() ? a[-1 - i] : 1;
    unsigned dimB = i < b.rank() ? b[-1 - i] : 1;
    unsigned dimC;

    if (dimA == dimB) {
      dimC = dimA;
    } else if (dimA == 1) {
      dimC = dimB;
    } else if (dimB == 1) {
      dimC = dimA;
    } else {
      throw BroadcastError(a, b, -1 - i);
    }
    dims_c[j] = dimC;

    strides_a[j] = strides_a[j + 1] * dimA;
    strides_b[j] = strides_b[j + 1] * dimB;
    strides_c[j] = strides_c[j + 1] * dimC;
  }

  for (int i = 0; i < dims_c.size(); i++) {
    int j = (int) dims_c.size() - 1 - i;
    unsigned dimA = i < a.rank() ? a[-1 - i] : 1;
    unsigned dimB = i < b.rank() ? b[-1 - i] : 1;

    if (dimA == dimB) {
    } else if (dimA == 1) {
      strides_a[j + 1] = 0;
    } else if (dimB == 1) {
      strides_b[j + 1] = 0;
    }
  }
}

}