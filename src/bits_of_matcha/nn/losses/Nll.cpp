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

  tensor likelihood;
  if (sparse && categorical) {
    likelihood = gather(predicted, expected, -1);
  } else if (sparse && !categorical) {
    likelihood = (expected == 1) * predicted + (expected == 0) * (1 - predicted);
  } else {
    // TODO: regressions, dense classification

    throw std::runtime_error("not implemented yet");
  }

  return -log(likelihood);
}

}