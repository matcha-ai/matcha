#pragma once

#include "bits_of_matcha/engine/memory/Buffer.h"
#include "bits_of_matcha/engine/iterations/MatrixwiseUnaryCtx.h"


namespace matcha::engine::cpu {

void transpose(engine::Buffer* a, engine::Buffer* b, const MatrixwiseUnaryCtx& ctx);

}