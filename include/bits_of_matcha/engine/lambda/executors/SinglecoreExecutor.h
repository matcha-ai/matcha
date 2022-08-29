#pragma once

#include "bits_of_matcha/engine/lambda/Executor.h"

#include <vector>
#include <variant>

namespace matcha::engine {

struct SinglecoreExecutor : Executor {
  explicit SinglecoreExecutor(Lambda&& lambda);

  void runInternal() override;

private:
  std::vector<std::variant<Op*, Tensor*>> instructions_;
};


}