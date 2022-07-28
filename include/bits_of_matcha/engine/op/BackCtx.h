#pragma once

#include <vector>


namespace matcha::engine {

class Op;
class Tensor;

struct BackCtx {
  Op* forward;
  std::vector<Tensor*> vals;
  std::vector<bool> wrts;
};

struct BackOps {
  BackOps(Op* op);
  BackOps(const std::vector<Op*>& ops, const std::vector<Tensor*>& outputs);
  BackOps() = default;

  std::vector<Op*> ops;
  std::vector<Tensor*> outputs;
};

}