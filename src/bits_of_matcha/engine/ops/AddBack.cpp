#include "bits_of_matcha/engine/ops/AddBack.h"


namespace matcha::engine::ops {

AddBack::AddBack(const BackCtx& ctx)
  : OpBack(ctx)
{
}

OpMeta<AddBack> AddBack::meta {
  .name = "AddBack",
};

}