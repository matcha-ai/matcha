#include "bits_of_matcha/engine/ops/Add.h"
#include "bits_of_matcha/engine/ops/AddBack.h"


namespace matcha::engine::ops {

Add::Add(Tensor* a, Tensor* b)
 : ElementwiseBinaryOp(a, b)
{}

OpMeta<Add> Add::meta {
  .name = "Add",
  .back = [](auto ctx) { return new AddBack(ctx); }
};

void Add::run() {
  runCPU(std::plus<float>());
}

}