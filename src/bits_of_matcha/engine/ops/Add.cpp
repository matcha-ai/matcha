#include "bits_of_matcha/engine/ops/Add.h"


namespace matcha::engine::ops {

Add::Add(Tensor* a, Tensor* b)
 : ElementwiseBinaryOp(a, b)
{}

OpMeta<Add> Add::meta {
  .name = "Add",
  .back = [](auto ctx) { return new AddBack(ctx); }
};

void Add::run() {
  outputs[0]->malloc();
//  print("add run: ", outputs[0]->buffer());
  runCPU(std::plus<float>());
}


AddBack::AddBack(const BackCtx& ctx)
  : OpBack(ctx)
{
}

OpMeta<AddBack> AddBack::meta {
  .name = "AddBack",
};

}