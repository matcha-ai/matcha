#include "bits_of_matcha/engine/iterations/MatrixwiseBinaryCtx.h"
#include "bits_of_matcha/error/BroadcastError.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

MatrixwiseBinaryCtx::MatrixwiseBinaryCtx(const Shape& a, const Shape& b) {
  if (std::min(a.rank(), b.rank()) < 2) {
    throw std::invalid_argument("both inputs must have rank at least 2");
  }

  prefixDimsC.resize(std::max(a.rank(), b.rank()) - 2);
  prefixStridesA.resize(prefixDimsC.size() + 1);
  prefixStridesB.resize(prefixDimsC.size() + 1);
  prefixStridesC.resize(prefixDimsC.size() + 1);

  colsA = a[-1];
  colsB = b[-1];
  rowsA = a[-2];
  rowsB = b[-2];

  prefixStridesA[prefixStridesA.size() - 1] = 1;
  prefixStridesB[prefixStridesB.size() - 1] = 1;
  prefixStridesC[prefixStridesC.size() - 1] = 1;

  for (int i = 0; i < prefixDimsC.size(); i++) {
    int j = i + 2;
    unsigned dimA = j < a.rank() ? a[-1 - j] : 1;
    unsigned dimB = j < b.rank() ? b[-1 - j] : 1;
    unsigned dimC;

    if (dimA == dimB) {
      dimC = dimA;
    } else if (dimA == 1) {
      dimC = dimB;
    } else if (dimB == 1) {
      dimC = dimA;
    } else {
      throw BroadcastError(a, b);
    }

    prefixDimsC[prefixDimsC.size() - 1 - i] = dimC;
    prefixStridesA[prefixStridesA.size() - 2 -i ] = dimA * prefixStridesA[prefixStridesA.size() - 1 - i];
    prefixStridesB[prefixStridesB.size() - 2 -i ] = dimB * prefixStridesB[prefixStridesB.size() - 1 - i];
    prefixStridesC[prefixStridesC.size() - 2 -i ] = dimC * prefixStridesC[prefixStridesC.size() - 1 - i];
  }

  for (int i = 0; i < prefixDimsC.size(); i++) {
    int j = i + 2;
    unsigned dimA = j < a.rank() ? a[-1 - j] : 1;
    unsigned dimB = j < b.rank() ? b[-1 - j] : 1;

    if (dimA == 1) {
      prefixStridesA[prefixStridesA.size() - 1 - i] = 0;
    }
    if (dimB == 1) {
      prefixStridesB[prefixStridesB.size() - 1 - i] = 0;
    }
  }

//  for (int i = 0 ; i < prefixStridesA.size(); i++) {
//    print(prefixStridesA[i], " ", prefixStridesB[i], " -> ", prefixStridesC[i]);
//  }
//  print("---");
//  exit(0);
}

}