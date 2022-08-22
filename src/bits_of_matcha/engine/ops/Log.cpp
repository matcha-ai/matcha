#include "bits_of_matcha/engine/ops/Log.h"
#include "bits_of_matcha/engine/ops/Divide.h"
#include "bits_of_matcha/engine/ops/Multiply.h"
#include "bits_of_matcha/engine/tensor/factories.h"


namespace matcha::engine::ops {

Log::Log(Tensor* a)
  : ElementwiseUnaryOp(a, promoteDtypes(a->dtype(), Float))
{}

Reflection<Log> Log::reflection {
  .name = "Log",
  .back = [](const BackCtx& ctx){
    // y = log a
    // dy/da = 1/a * da
    auto one = engine::ones({});
    auto temp = dispatch<Divide>(one, ctx.forward->inputs[0])[0];
    return dispatch<Multiply>(temp, ctx.vals[0]);
  },
};

void Log::run() {
  runCpuReal([](auto a) { return std::log(a); });
}

}