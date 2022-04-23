#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/autograd/OpBack.h"


namespace matcha::engine::ops {

struct Identity : Op {
  Identity(Tensor* a);
  static OpMeta<Identity> meta;

  void run() override;
};

struct IdentityBack : OpBack {
  IdentityBack(const BackCtx& ctx);
  static OpMeta<IdentityBack> meta;

  void run() override;
};

}
