#pragma once

#include "bits_of_matcha/engine/memory/Block.h"
#include "bits_of_matcha/engine/iterations/MatrixwiseBinaryCtx.h"

#include <cblas.h>


namespace matcha::engine::cpu {

template <class T>
void mm(Buffer& a, Buffer& b, Buffer& c, const MatrixwiseBinaryCtx& ctx) {
  size_t sizeA = ctx.rows_a * ctx.cols_a;
  size_t sizeB = ctx.rows_b * ctx.cols_b;
  size_t sizeC = ctx.rows_a * ctx.cols_b;

  CBLAS_TRANSPOSE trans_a, trans_b;
  switch (ctx.transpose.first) {
  case 'N': trans_a = CblasNoTrans; break;
  case 'T': trans_a = CblasTrans; break;
  case 'H': trans_a = CblasConjTrans; break;
  }
  switch (ctx.transpose.second) {
  case 'N': trans_b = CblasNoTrans; break;
  case 'T': trans_b = CblasTrans; break;
  case 'H': trans_b = CblasConjTrans; break;
  }

  auto vals_a = a.as<T*>();
  auto vals_b = b.as<T*>();
  auto vals_c = c.as<T*>();

  int strides = (int) ctx.prefix_strides_a.size();

  std::vector<unsigned> begin_a(strides, 0);
  std::vector<unsigned> begin_b(strides, 0);
  std::vector<unsigned> begin_c(strides, 0);
  auto iter_c = vals_c;

  int counter = 0;
  int axis = 1;
  while (true) {
//    if (counter++ > 20) exit(69);
    if (axis < strides - 1) {
      print("==========");
      print("axis ", axis, " begin_c ", begin_c[axis], " strides_c ", ctx.prefix_strides_c[axis - 1]);
//    print("axis: ", axis, " (strides ", strides, ")");
//      print("strides: ", ctx.prefix_strides_a[axis], " ", ctx.prefix_strides_b[axis], " -> ", ctx.prefix_strides_c[axis]);
//      print("begins: ", begin_a[axis], " ", begin_b[axis], " -> ", begin_c[axis]);
//      print("BC: ", begin_c[axis], " ", begin_c[axis - 1] + ctx.prefix_strides_c[axis - 1]);
//      print("begin_c ", begin_c[axis]);
      if (begin_c[axis] == begin_c[axis - 1] + ctx.prefix_strides_c[axis - 1]) {
//        print("-> dec axis");
        if (axis == 1) break;
        axis--;
      } else {
        begin_a[axis + 1] = begin_a[axis];
        begin_b[axis + 1] = begin_b[axis];
        begin_c[axis + 1] = begin_c[axis];
        begin_a[axis] += ctx.prefix_strides_a[axis];
        begin_b[axis] += ctx.prefix_strides_b[axis];
        begin_c[axis] += ctx.prefix_strides_c[axis];
//        print("-> inc axis");
        axis++;
      }
    } else {
      if (axis >= strides) {
        axis = strides - 1;
      }

//      print(ctx.prefix_strides_c[axis - 1]);
//      print(begin_a[axis]);
      auto matA = vals_a + begin_a[axis] * sizeA;
      auto matB = vals_b + begin_b[axis] * sizeB;
      auto loops = axis - 1 >= 0 ? ctx.prefix_strides_c[axis - 1] : 1;
      for (auto itC = 0; itC != loops; itC++) {
        auto matC = vals_c + (begin_c[axis] + itC) * sizeC;

//        print(matA - vals_a, " ", matB - vals_b, " -> ", matC - vals_c);
///*
        if constexpr (std::is_same<T, float>()) {
          cblas_sgemm(
            CblasRowMajor,
            trans_a,
            trans_b,
            (int) ctx.rows_a,
            (int) ctx.cols_b,
            (int) ctx.cols_a,
            1,
            matA,
            (int) ctx.cols_a,
            matB,
            (int) ctx.cols_b,
            0,
            matC,
            (int) ctx.cols_b
          );
        } else if constexpr (std::is_same<T, double>()) {
          cblas_dgemm(
            CblasRowMajor,
            trans_a,
            trans_b,
            (int) ctx.rows_a,
            (int) ctx.cols_b,
            (int) ctx.cols_a,
            1,
            matA,
            (int) ctx.cols_a,
            matB,
            (int) ctx.cols_b,
            0,
            matC,
            (int) ctx.cols_b
          );

        } else if constexpr (std::is_same<T, std::complex<float>>()) {
          cblas_cgemm(
            CblasRowMajor,
            trans_a,
            trans_b,
            (int) ctx.rows_a,
            (int) ctx.cols_b,
            (int) ctx.cols_a,
            1,
            matA,
            (int) ctx.cols_a,
            matB,
            (int) ctx.cols_b,
            0,
            matC,
            (int) ctx.cols_b
          );

        } else if constexpr (std::is_same<T, std::complex<double>>()) {
          cblas_zgemm(
            CblasRowMajor,
            trans_a,
            trans_b,
            (int) ctx.rows_a,
            (int) ctx.cols_b,
            (int) ctx.cols_a,
            1,
            matA,
            (int) ctx.cols_a,
            matB,
            (int) ctx.cols_b,
            0,
            matC,
            (int) ctx.cols_b
          );

        }
//        */

        matA += sizeA * ctx.prefix_strides_a[axis];
        matB += sizeB * ctx.prefix_strides_b[axis];
      }
//      print("blas");

      if (axis <= 1) break;
      axis--;
    }

  }


}

}

