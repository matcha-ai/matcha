#pragma once

#include "bits_of_matcha/engine/chain/Pass.h"

#include <vector>
#include <memory>
#include <set>
#include <iostream>

namespace matcha::engine {

class Tensor;
class Op;

struct Chain {
  std::vector<Tensor*> tensors;
  std::vector<Tensor*> inputs;
  std::vector<Tensor*> outputs;
  std::vector<Op*> ops;

  Chain() = default;
  Chain(const Chain& other) = delete;
  Chain(Chain&& other) noexcept = default;

  Chain& operator=(const Chain& other) = delete;
  Chain& operator=(Chain&& other) noexcept = default;

  ~Chain();
};

Chain clone(const Chain& chain);
std::ostream& operator<<(std::ostream& os, const Chain& chain);

}