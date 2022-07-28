#include "bits_of_matcha/engine/op/BackCtx.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

BackOps::BackOps(Op* op)
: ops {op} {
  outputs = op->outputs.stdVector();
}

BackOps::BackOps(const std::vector<Op*>& ops, const std::vector<Tensor*>& outputs)
: ops {ops}, outputs(outputs) {}


}