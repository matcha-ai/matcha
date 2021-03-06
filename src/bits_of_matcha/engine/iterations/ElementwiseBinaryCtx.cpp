#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"
#include "bits_of_matcha/error/BroadcastError.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

ElementwiseBinaryCtx::ElementwiseBinaryCtx(const Shape& a, const Shape& b)
  : dimsC(std::max(a.rank(), b.rank()))
  , stridesA(dimsC.size() + 1)
  , stridesB(dimsC.size() + 1)
  , stridesC(dimsC.size() + 1)
{
  stridesA[stridesA.size() - 1] = 1;
  stridesB[stridesB.size() - 1] = 1;
  stridesC[stridesC.size() - 1] = 1;

  for (int i = 0; i < dimsC.size(); i++) {
    int j = (int) dimsC.size() - 1 - i;
    unsigned dimA = i < a.rank() ? a[-1 - i] : 1;
    unsigned dimB = i < b.rank() ? b[-1 - i] : 1;
    unsigned dimC;

    if (dimA == dimB) {
      dimC = dimA;
    } else if (dimA == 1) {
      dimC = dimB;
    } else if (dimB == 1) {
      dimC = dimA;
    } else {
      throw BroadcastError(a, b, -1 - i);
    }
    dimsC[j] = dimC;

    stridesA[j] = stridesA[j + 1] * dimA;
    stridesB[j] = stridesB[j + 1] * dimB;
    stridesC[j] = stridesC[j + 1] * dimC;
  }

  for (int i = 0; i < dimsC.size(); i++) {
    int j = (int) dimsC.size() - 1 - i;
    unsigned dimA = i < a.rank() ? a[-1 - i] : 1;
    unsigned dimB = i < b.rank() ? b[-1 - i] : 1;

    if (dimA == dimB) {
    } else if (dimA == 1) {
      stridesA[j + 1] = 0;
    } else if (dimB == 1) {
      stridesB[j + 1] = 0;
    }
  }
}

}