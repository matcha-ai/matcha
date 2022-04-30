#pragma once

#include "bits_of_matcha/engine/memory/Buffer.h"
#include "bits_of_matcha/engine/iterations/AxiswiseFoldCtx.h"


namespace matcha::engine::cpu {


template <class Callable>
void fold(Callable callable, float init, engine::Buffer* a, engine::Buffer* b, const FoldCtx& ctx) {

}

}