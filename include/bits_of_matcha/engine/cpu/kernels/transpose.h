#pragma once

#include "bits_of_matcha/engine/memory/Block.h"
#include "bits_of_matcha/engine/iterations/MatrixwiseUnaryCtx.h"


namespace matcha::engine::cpu {

template <class T = float>
void transpose(Buffer& a, Buffer& b, const MatrixwiseUnaryCtx& ctx) {
  auto valsA = a.as<T*>();
  auto valsB = b.as<T*>();

  auto matBeginA = valsA;
  auto matBeginB = valsB;
  auto iterB = matBeginB;
  for (int mat = 0; mat < ctx.mats; mat++) {

    for (T* colA = matBeginA; colA != matBeginA + ctx.cols; colA++) {
      for (T* rowA = colA; rowA != colA + ctx.size; rowA += ctx.cols) {
        *iterB++ = *rowA;
      }
    }

    matBeginA += ctx.size;
//    matBeginB += ctx.size;
  }
}

}