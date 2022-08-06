#pragma once

#include "bits_of_matcha/engine/chain/Pass.h"

#include <vector>
#include <memory>
#include <set>
#include <iostream>
#include <map>

namespace matcha {
class tensor;
}

namespace matcha::engine {

class Tensor;
class Op;

struct Chain final {
  std::vector<Tensor*> tensors;
  std::vector<Tensor*> inputs;
  std::vector<Tensor*> outputs;
  std::vector<Op*> ops;
  std::map<Tensor*, const tensor*> side_inputs;

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