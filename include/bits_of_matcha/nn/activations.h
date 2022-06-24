#include "matcha/tensor"


namespace matcha::nn {

tensor relu(const tensor& a) { return maxBetween(a, 0); }

struct Relu {
  tensor operator()(const tensor& a) { return relu(a); }
};

}