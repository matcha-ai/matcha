#include "bits_of_matcha/engine/iterations/AxiswiseFoldCtx.h"
#include "bits_of_matcha/print.h"

namespace matcha::engine {

AxiswiseFoldCtx::AxiswiseFoldCtx(const Shape& a, bool keep_dims) {
  axis_stride = 1;
  axis_length = a.size();
  prefix_strides = {0, 0};
  prefix_lengths = {1, 1};
//  exit(0);
}

AxiswiseFoldCtx::AxiswiseFoldCtx(const Shape& a, int axis, bool keep_dims) {
  if (axis < 0) axis += (int) a.rank();
  if (axis < 0 || axis >= a.rank()) throw std::invalid_argument("axis is out of range");

//  axis_length = a[axis];
  prefix_strides = prefix_lengths = {1, 1};
  axis_length = axis_stride = 1;
  if (axis == a.rank() - 1) {
//    axis_stride = 1;
//    prefix_strides = {1, a[-1]};
//    prefix_lengths = {1, a.size() / prefix_strides[1]};

  } else {

  }

  size_t dimBuff = 1;
  for (int i = (int) a.rank() - 1; i >= 0; i--) {
    auto dim = a[i];

//    print("i ", i, " dimBuff ", dimBuff);
    if (i == axis) {
//      print("axis");
      axis_stride = dimBuff;
      axis_length = dim;

      if (axis == a.rank() - 1) {
        prefix_strides[1] = a[-1];
        prefix_lengths[1] = a.size() / a[-1];

        prefix_strides[0] = 0;
        prefix_lengths[0] = 1;
      } else if (axis == 0) {
        prefix_strides[1] = 1;
        prefix_lengths[1] = dimBuff;

        prefix_strides[0] = 0;
        prefix_lengths[0] = 1;
      } else {
        prefix_strides[1] = 1;
        prefix_lengths[1] = dimBuff;

        prefix_strides[0] = dimBuff * dim;
        prefix_lengths[0] = a.size() / prefix_strides[0];
      }
    }
    dimBuff *= dim;
  }

//  print("axis: ", axis_stride, " ", axis_length);
//  print("0: ", prefix_strides[0], " ", prefix_lengths[0]);
//  print("1: ", prefix_strides[1], " ", prefix_lengths[1]);
//  exit(0);
}

}