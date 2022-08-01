#pragma once

#include "bits_of_matcha/engine/chain/Executor.h"

namespace matcha::engine {

struct SinglecoreExecutor : Executor {
  explicit SinglecoreExecutor(Chain&& chain);

  void runInternal() override;
};


}