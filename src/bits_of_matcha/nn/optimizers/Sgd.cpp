#include "bits_of_matcha/nn/optimizers/Sgd.h"


namespace matcha::nn {

void Sgd::operator()(std::map<tensor*, tensor>& gradients) const {
  constexpr bool debug = false;

  for (auto&& [t, g]: gradients) {
    if constexpr (debug) {
      std::cerr << "target:" << std::endl;
      std::cerr << t << std::endl;
      std::cerr << std::endl << std::endl << std::endl;
      std::cerr << "grad:" << std::endl;
      std::cerr << g << std::endl;
      std::cerr << std::endl << std::endl << std::endl;
      std::cerr << std::endl << std::endl << std::endl;
    }
    *t -= lr * g;
  }
}

}