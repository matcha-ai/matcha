#pragma once

#include <vector>


namespace matcha::engine {

class Op;
class Tensor;

struct BackCtx {
  Op* forward;
  std::vector<Tensor*> vals;
  std::vector<Tensor*> wrts;
};

}