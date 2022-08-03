#include "bits_of_matcha/nn/activations.h"


namespace matcha::nn {

tensor tanh(const tensor& batch) {
  return matcha::tanh(batch);
}

tensor sigmoid(const tensor& batch) {
  return matcha::sigmoid(batch);
}

tensor softmax(const tensor& batch) {
  // batch-wise softmax
  return matcha::softmax(batch, 0);
}

tensor relu(const tensor& batch) {
  return maximum(batch, cast(0, batch.dtype()));
}

}