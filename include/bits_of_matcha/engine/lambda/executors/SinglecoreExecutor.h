#pragma once

#include "bits_of_matcha/engine/lambda/Executor.h"

namespace matcha::engine {

struct SinglecoreExecutor : Executor {
  explicit SinglecoreExecutor(Lambda&& lambda);

  void runInternal() override;
};


}