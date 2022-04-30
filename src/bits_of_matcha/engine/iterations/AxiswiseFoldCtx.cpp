#include "bits_of_matcha/engine/iterations/AxiswiseCtx.h"
#include "bits_of_matcha/print.h"

namespace matcha::engine {

AxiswiseCtx::AxiswiseCtx(const Shape& a) {

  exit(0);
}

AxiswiseCtx::AxiswiseCtx(const Shape& a, int axis) {
  if (axis < 0) axis += (int) a.rank();
  if (axis < 0 || axis >= a.rank()) throw std::invalid_argument("axis is out of range");

//  axisLength = a[axis];
  prefixStrides = prefixLengths = {1, 1};
  axisLength = axisStride = 1;
  if (axis == a.rank() - 1) {
//    axisStride = 1;
//    prefixStrides = {1, a[-1]};
//    prefixLengths = {1, a.size() / prefixStrides[1]};

  } else {

  }

  size_t dimBuff = 1;
  for (int i = (int) a.rank() - 1; i >= 0; i--) {
    auto dim = a[i];

//    print("i ", i, " dimBuff ", dimBuff);
    if (i == axis) {
//      print("axis");
      axisStride = dimBuff;
      axisLength = dim;

      if (axis == a.rank() - 1) {
        prefixStrides[1] = a[-1];
        prefixLengths[1] = a.size() / a[-1];

        prefixStrides[0] = 0;
        prefixLengths[0] = 1;
      } else if (axis == 0) {
        prefixStrides[1] = 1;
        prefixLengths[1] = dimBuff;

        prefixStrides[0] = 0;
        prefixLengths[0] = 1;
      } else {

      }
    }
    dimBuff *= dim;
  }

  print("axis: ", axisStride, " ", axisLength);
  print("0: ", prefixStrides[0], " ", prefixLengths[0]);
  print("1: ", prefixStrides[1], " ", prefixLengths[1]);
  exit(0);
}

}