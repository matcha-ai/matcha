#include "bits_of_matcha/Engine.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/dataset/Dataset.h"
#include "bits_of_matcha/dataset/InstanceIterator.h"
#include "bits_of_matcha/engine/dataset/Dataset.h"
#include "bits_of_matcha/print.h"


namespace matcha {

using engine::Tensor;

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

Dataset Engine::ref(engine::Dataset* internal) {
  return Dataset(internal);
}

engine::Dataset* Engine::deref(const Dataset* external) {
  return (engine::Dataset*) external->internal_;
}

engine::Dataset* Engine::deref(const Dataset& external) {
  return deref(&external);
}

engine::Dataset* Engine::unref(Dataset* external) {
  auto internal = deref(external);
  external->internal_ = nullptr;
  return internal;
}

engine::Dataset* Engine::unref(Dataset& external) {
  return unref(&external);
}

InstanceIterator Engine::makeInstanceIterator(engine::Dataset* ds, size_t pos) {
  return InstanceIterator(ds, pos);
}


}