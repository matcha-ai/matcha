#pragma once

#include "bits_of_matcha/engine/lambda/Pass.h"

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

struct Lambda final {
  std::vector<Tensor*> tensors;
  std::vector<Tensor*> inputs;
  std::vector<Tensor*> outputs;
  std::vector<Op*> ops;
  std::map<Tensor*, const tensor*> side_inputs;

  Lambda() = default;
  Lambda(const Lambda& other);
  Lambda(Lambda&& other) noexcept = default;

  Lambda& operator=(const Lambda& other);
  Lambda& operator=(Lambda&& other) noexcept = default;

  operator bool() const;

  ~Lambda();
};

Lambda clone(const Lambda& lambda);
std::ostream& operator<<(std::ostream& os, const Lambda& lambda);

}