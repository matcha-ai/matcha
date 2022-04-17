#pragma once

#include <vector>


namespace matcha::engine {

class Tensor;
class OpBack;
class Op;

/**
 * context for adjoint backpropagation op:
 * forward op and partials
 */
struct BackCtx {
  Op* forward;
  std::vector<Tensor*> vals;
  std::vector<Tensor*> wrts;
};

}