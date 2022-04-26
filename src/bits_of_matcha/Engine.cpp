#include "bits_of_matcha/Engine.h"
#include "bits_of_matcha/tensor.h"


namespace matcha {

using namespace engine;

tensor Engine::ref(Tensor* internal) {
  return tensor(internal);
}

Tensor* Engine::unref(tensor& external) {
  return unref(&external);
}

Tensor* Engine::unref(tensor* external) {
  auto t = (Tensor*) external->internal_;
  external->internal_ = nullptr;
  return t;
}

Tensor* Engine::deref(const tensor* external) {
  return (Tensor*) external->internal_;
}

Tensor* Engine::deref(const tensor& external) {
  return deref(&external);
}


}