#include "matcha/tensor"


namespace matcha::nn {

tensor tanh(const tensor& batch);
tensor sigmoid(const tensor& batch);
tensor softmax(const tensor& batch);

tensor relu(const tensor& batch);
//tensor elu(const tensor& batch);
//tensor gelu(const tensor& batch);

struct Tanh { tensor operator()(const tensor& batch) { return nn::tanh(batch); } };
struct Sigmoid { tensor operator()(const tensor& batch) { return nn::sigmoid(batch);} };
struct Softmax { tensor operator()(const tensor& batch) { return nn::softmax(batch);} };

struct Relu { tensor operator()(const tensor& batch) { return relu(batch);} };
//struct Elu { tensor operator()(const tensor& batch) { return elu(batch); } };
//struct Gelu { tensor operator()(const tensor& batch) { return gelu(batch); } };


}