#pragma once

#include "bits_of_matcha/engine/autograd/OpBack.h"


namespace matcha::engine::ops {

struct DotBack : OpBack {
  DotBack(const BackCtx& ctx);
  static OpMeta<DotBack> meta;

  void run() override;
};

}
