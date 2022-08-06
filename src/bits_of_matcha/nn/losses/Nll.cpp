#include "bits_of_matcha/nn/losses/Nll.h"


namespace matcha::nn {

tensor Nll::operator()(const tensor& expected, const tensor& predicted) const {
  bool sparse;
  bool categorical;

  switch (expected.dtype()) {
  case Half:
  case Float:
  case Double:
  case Cfloat:
  case Cdouble:
    sparse = false;
    break;
  default:
    sparse = true;
    break;
  }

  categorical = predicted.size() > 0;

  // TODO: regressions

  if (sparse && categorical) {
    tensor p = predicted[expected];
    return -log(p);
  }

  return 0;
}

}