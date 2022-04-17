#pragma once

#include "bits_of_matcha/engine/ops/Add.h"
#include "bits_of_matcha/engine/autograd/OpBack.h"


namespace matcha::engine::ops {


struct AddBack : OpBack {
  AddBack(const BackCtx& ctx);
  static OpMeta<AddBack> meta;

};


}