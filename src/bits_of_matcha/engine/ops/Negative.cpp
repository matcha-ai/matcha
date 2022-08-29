#include "bits_of_matcha/engine/ops/Negative.h"
#include "bits_of_matcha/engine/ops/Multiply.h"

#include <cmath>


namespace matcha::engine::ops {

Negative::Negative(Tensor* a)
: ElementwiseUnaryOp(a, promoteDtypes(a->dtype(), Byte))
{}

Reflection<Negative> Negative::reflection {
  .name = "Negative",
  .back = [](const BackCtx& ctx) {
    return dispatch<Negative>(ctx.vals[0]);
  },
};

void Negative::run() {
  runCpu([](auto x) { return -x; });
}

}
