#include "bits_of_matcha/nn/layers/Flatten.h"


namespace matcha::nn {

tensor Flatten::operator()(const tensor& a) {
  if (a.rank() < 2) throw std::invalid_argument("Flatten input must have shape [batchSize, dims...]");
  unsigned batch = a.shape()[0];
  return a.reshape(batch, -1);
}

}
