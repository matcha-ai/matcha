#include "bits_of_matcha/engine/ops/Max.h"

#include <limits>
#include <execution>
#include <algorithm>

namespace matcha::engine::ops {

Max::Max(Tensor* a)
  : AxiswiseFoldOp(a)
{}

Max::Max(Tensor* a, int axis)
  : AxiswiseFoldOp(a, axis)
{}

void Max::run() {
  outputs[0]->malloc();

  runCPU(
    [](float* begin, size_t stride, float* end) {
      float buffer = std::numeric_limits<float>::min();
      if (stride != 1) {
        for (float* iter = begin; iter != end; iter += stride) {
          buffer = std::max(buffer, *iter);
        }
      } else {
        auto result = std::max_element(std::execution::par_unseq, begin, end);
        buffer = *result;
      }

      return buffer;
    }
  );
}

}