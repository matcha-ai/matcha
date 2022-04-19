#include "bits_of_matcha/engine/ops/TransposeBack.h"


namespace matcha::engine::ops {

TransposeBack::TransposeBack(const BackCtx& ctx)
: OpBack(ctx)
{}

OpMeta<TransposeBack> TransposeBack::meta {
.name = "TransposeBack",
};

void TransposeBack::run() {

}

}
