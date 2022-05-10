#include "bits_of_matcha/nn/layers/Activation.h"

#include <algorithm>


namespace matcha::nn {

Activation::Activation(const char* activation)
  : Activation(std::string(activation))
{}

Activation::Activation(std::string activation) {
  std::transform(
    activation.begin(), activation.end(),
    activation.begin(),
    tolower
  );

  if (activation.empty() || activation == "none" || activation == "identity") {
    internal_ = identity;
    return;
  }

  if (activation == "relu") {
    internal_ = [](const tensor& a) { return maxBetween(a, 0); };
    return;
  }
}

Activation::Activation(const UnaryOp& activation)
  : internal_(activation)
{}

tensor Activation::operator()(const tensor& a) {
  return internal_(a);
}

}