#include "bits_of_matcha/engine/ops/Argmax.h"

#include <limits>
#include <execution>
#include <algorithm>

namespace matcha::engine::ops {

Argmax::Argmax(Tensor* a)
  : AxiswiseFoldOp(a)
{}

Argmax::Argmax(Tensor* a, int axis)
  : AxiswiseFoldOp(a, axis)
{}

OpMeta<Argmax> Argmax::meta {
  .name = "Argmax",
};

void Argmax::run() {
  outputs[0]->malloc();

  runCPU<float>(
    [](float* begin, size_t stride, float* end) {
      float buffer = std::numeric_limits<float>::min();
      float* pos;
      if (stride != 1) {
        for (float* iter = begin; iter != end; iter += stride) {
          if (*iter > buffer) {
            buffer = *iter;
            pos = iter;
          }
        }
      } else {
        auto result = std::max_element(std::execution::par_unseq, begin, end);
        pos = &*result;
      }

      return (float) ((pos - begin) / stride);
    }
  );
}

}
