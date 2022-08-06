#include "bits_of_matcha/Engine.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/dataset/Dataset.h"
#include "bits_of_matcha/dataset/InstanceIterator.h"
#include "bits_of_matcha/engine/dataset/Dataset.h"
#include "bits_of_matcha/print.h"
#include "bits_of_matcha/fn.h"
#include "bits_of_matcha/View.h"
#include "bits_of_matcha/engine/tensor/Binding.h"
#include "bits_of_matcha/engine/chain/Tracer.h"


namespace matcha {

using engine::Tensor;

tensor Engine::ref(Tensor* internal) {
  return tensor(internal);
}

Tensor* Engine::unref(tensor& external) {
  return unref(&external);
}

Tensor* Engine::unref(tensor* external) {
  auto&& internal = (engine::Binding*) external->internal_;
  Tensor* t = nullptr;
  if (internal) {
    t = internal->get();
    internal->unref();
  }
  external->internal_ = nullptr;

  engine::Tracer::handleNewDeref(external, t);
  return t;
}

Tensor* Engine::deref(const tensor* external) {
  auto&& binding = (engine::Binding*) external->internal_;
  if (!binding) return nullptr;
  auto&& internal = binding->get();
  engine::Tracer::handleNewDeref(external, internal);
  return internal;
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

matcha::View Engine::ref(engine::View* internal) {
  return matcha::View(internal);
}

engine::View* Engine::deref(const matcha::View& external) {
  return (engine::View*) external.internal_;
}


}