#include "bits_of_matcha/engine/iterations/MatrixwiseBinaryCtx.h"
#include "bits_of_matcha/error/BroadcastError.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

MatrixwiseBinaryCtx::MatrixwiseBinaryCtx(const Shape& a, const Shape& b,
                                         std::pair<char, char> transpose)
  : transpose(transpose)
{
  if (std::min(a.rank(), b.rank()) < 2) {
    throw std::invalid_argument("both inputs must have rank at least 2");
  }

  prefix_dims_c.resize(std::max(a.rank(), b.rank()) - 2);
  prefix_strides_a.resize(prefix_dims_c.size() + 1);
  prefix_strides_b.resize(prefix_dims_c.size() + 1);
  prefix_strides_c.resize(prefix_dims_c.size() + 1);

  cols_a = a[-1];
  cols_b = b[-1];
  rows_a = a[-2];
  rows_b = b[-2];

  if (transpose.first != 'N') std::swap(cols_a, rows_a);
  if (transpose.second != 'N') std::swap(cols_b, rows_b);

  prefix_strides_a[prefix_strides_a.size() - 1] = 1;
  prefix_strides_b[prefix_strides_b.size() - 1] = 1;
  prefix_strides_c[prefix_strides_c.size() - 1] = 1;

  for (int i = 0; i < prefix_dims_c.size(); i++) {
    int j = i + 2;
    unsigned dimA = j < a.rank() ? a[-1 - j] : 1;
    unsigned dimB = j < b.rank() ? b[-1 - j] : 1;
    unsigned dimC;

    if (dimA == dimB) {
      dimC = dimA;
    } else if (dimA == 1) {
      dimC = dimB;
    } else if (dimB == 1) {
      dimC = dimA;
    } else {
      throw BroadcastError(a, b);
    }

    prefix_dims_c[prefix_dims_c.size() - 1 - i] = dimC;
    prefix_strides_a[prefix_strides_a.size() - 2 - i ] = dimA * prefix_strides_a[prefix_strides_a.size() - 1 - i];
    prefix_strides_b[prefix_strides_b.size() - 2 - i ] = dimB * prefix_strides_b[prefix_strides_b.size() - 1 - i];
    prefix_strides_c[prefix_strides_c.size() - 2 - i ] = dimC * prefix_strides_c[prefix_strides_c.size() - 1 - i];
  }

  for (int i = 0; i < prefix_dims_c.size(); i++) {
    int j = i + 2;
    unsigned dimA = j < a.rank() ? a[-1 - j] : 1;
    unsigned dimB = j < b.rank() ? b[-1 - j] : 1;

    if (dimA == 1) {
      prefix_strides_a[prefix_strides_a.size() - 1 - i] = 0;
    }
    if (dimB == 1) {
      prefix_strides_b[prefix_strides_b.size() - 1 - i] = 0;
    }
  }
}

}