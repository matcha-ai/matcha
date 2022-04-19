#include "bits_of_matcha/engine/ops/DotBack.h"


namespace matcha::engine::ops {

DotBack::DotBack(const BackCtx& ctx)
  : OpBack(ctx)
{}

OpMeta<DotBack> DotBack::meta {
  .name = "DotBack",
};

void DotBack::run() {

}

}