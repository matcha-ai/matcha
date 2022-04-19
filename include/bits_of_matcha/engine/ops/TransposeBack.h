#pragma once

#include "bits_of_matcha/engine/autograd/OpBack.h"


namespace matcha::engine::ops {

struct TransposeBack : OpBack {
  TransposeBack(const BackCtx& ctx);
  static OpMeta<TransposeBack> meta;

  void run() override;
};

}
