#include "bits_of_matcha/nn/layers/Flatten.h"


namespace matcha::nn {

tensor flatten(const tensor& batch) {
  if (batch.rank() < 2) throw std::invalid_argument("Flatten input must have shape [batchSize, dims...]");
  unsigned bsize = batch.shape()[0];
  return batch.reshape(bsize, -1);
}

}
