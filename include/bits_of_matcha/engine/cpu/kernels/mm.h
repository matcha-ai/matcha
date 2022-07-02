#pragma once

#include "bits_of_matcha/engine/memory/Block.h"
#include "bits_of_matcha/engine/iterations/MatrixwiseBinaryCtx.h"

#include <cblas.h>


namespace matcha::engine::cpu {

void mm(Buffer& a, Buffer& b, Buffer& c, const MatrixwiseBinaryCtx& ctx) {
  size_t sizeA = ctx.rowsA * ctx.colsA;
  size_t sizeB = ctx.rowsB * ctx.colsB;
  size_t sizeC = ctx.rowsA * ctx.colsB;

  auto valsA = a.as<float*>();
  auto valsB = b.as<float*>();
  auto valsC = c.as<float*>();

  int strides = (int) ctx.prefixStridesA.size();

  std::vector<unsigned> beginA(strides, 0);
  std::vector<unsigned> beginB(strides, 0);
  std::vector<unsigned> beginC(strides, 0);
  auto iterC = valsC;

  int counter = 0;
  int axis = 1;
  while (true) {
//    if (counter++ > 20) exit(69);
    if (axis < strides - 1) {
      print("==========");
      print("axis ", axis, " beginC ", beginC[axis], " stridesC ", ctx.prefixStridesC[axis - 1]);
//    print("axis: ", axis, " (strides ", strides, ")");
//      print("strides: ", ctx.prefixStridesA[axis], " ", ctx.prefixStridesB[axis], " -> ", ctx.prefixStridesC[axis]);
//      print("begins: ", beginA[axis], " ", beginB[axis], " -> ", beginC[axis]);
//      print("BC: ", beginC[axis], " ", beginC[axis - 1] + ctx.prefixStridesC[axis - 1]);
//      print("beginC ", beginC[axis]);
      if (beginC[axis] == beginC[axis - 1] + ctx.prefixStridesC[axis - 1]) {
//        print("-> dec axis");
        if (axis == 1) break;
        axis--;
      } else {
        beginA[axis + 1] = beginA[axis];
        beginB[axis + 1] = beginB[axis];
        beginC[axis + 1] = beginC[axis];
        beginA[axis] += ctx.prefixStridesA[axis];
        beginB[axis] += ctx.prefixStridesB[axis];
        beginC[axis] += ctx.prefixStridesC[axis];
//        print("-> inc axis");
        axis++;
      }
    } else {
      if (axis >= strides) {
        axis = strides - 1;
      }

//      print(ctx.prefixStridesC[axis - 1]);
//      print(beginA[axis]);
      auto matA = valsA + beginA[axis] * sizeA;
      auto matB = valsB + beginB[axis] * sizeB;
      auto loops = axis - 1 >= 0 ? ctx.prefixStridesC[axis - 1] : 1;
      for (auto itC = 0; itC != loops; itC++) {
        auto matC = valsC + (beginC[axis] + itC) * sizeC;

//        print(matA - valsA, " ", matB - valsB, " -> ", matC - valsC);
///*
        cblas_sgemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasNoTrans,
          (int) ctx.rowsA,
          (int) ctx.colsB,
          (int) ctx.colsA,
          1,
          matA,
          (int) ctx.colsA,
          matB,
          (int) ctx.colsB,
          0,
          matC,
          (int) ctx.colsB
        );
//        */

        matA += sizeA * ctx.prefixStridesA[axis];
        matB += sizeB * ctx.prefixStridesB[axis];
      }
//      print("blas");

      if (axis <= 1) break;
      axis--;
    }

  }


}

}

