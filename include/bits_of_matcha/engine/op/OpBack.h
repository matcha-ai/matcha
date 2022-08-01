#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/op/BackCtx.h"


namespace matcha::engine {

struct OpBack : Op {
  OpBack(const BackCtx& ctx);

protected:
  Op* forward_;
};

}