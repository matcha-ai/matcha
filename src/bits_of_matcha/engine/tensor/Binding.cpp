#include "bits_of_matcha/engine/tensor/Binding.h"
#include "bits_of_matcha/Engine.h"
#include "bits_of_matcha/print.h"
#include "bits_of_matcha/View.h"
#include "bits_of_matcha/engine/lambda/Tracer.h"
#include "bits_of_matcha/engine/ops/ViewRead.h"
#include "bits_of_matcha/engine/ops/ViewWrite.h"


namespace matcha::engine {

Binding::Binding(Tensor* tensor)
  : tensor_(nullptr)
  , refs_(0)
{
  set(tensor);
}

void Binding::ref() {
  refs_++;
}

void Binding::unref() {
  if (!refs_)
    throw std::runtime_error("ref counter is already 0!");

  if (!--refs_)
    delete this;
}

Binding::~Binding() {
  set(nullptr);
}

unsigned Binding::refs() const {
  return refs_;
}

Tensor* Binding::get() {
  return tensor_;
}

void Binding::set(Tensor* tensor) {
  if (tensor_)
    tensor_->unreq();

  tensor_ = tensor;

  if (tensor_) {
    tensor_->req();
    Tracer::handleNewTensor(tensor);
  }

}

View::View(View* parent, engine::Tensor* idx)
  : parent_(parent)
  , idx_(idx)
  , binding_(parent->binding())
  , cache_(nullptr)
  , refs_(0)
{
  parent_->ref();
  idx->req();
}

View::View(Binding* binding, engine::Tensor* idx)
  : parent_(nullptr)
  , idx_(idx)
  , binding_(binding)
  , cache_(nullptr)
  , refs_(0)
{
  binding_->ref();
  idx->req();
}

void View::ref() {
  refs_++;
}

void View::unref() {
  if (!refs_)
    throw std::runtime_error("ref count is already 0!");

  if (!--refs_)
    delete this;
}

View::~View() {
  if (parent_)
    parent_->unref();
  else
    binding_->unref();

  if (cache_)
    cache_->unreq();

  idx_->unreq();
}

const Frame& View::frame() {
  return cache()->frame();
}

engine::Tensor* View::read() {
  return cache();
}

void View::write(engine::Tensor* rhs) {
  auto result = dispatch<ops::ViewWrite>(binding_->get(), rhs, indices())[0];
  binding()->set(result);
}

engine::Tensor* View::cache() {
  if (cache_) return cache_;
  cache_ = dispatch<ops::ViewRead>(binding()->get(), indices())[0];
  cache_->req();
  return cache_;
}

Binding* View::binding() {
  return binding_;
}

std::vector<Tensor*> View::indices() {
  if (!parent_)
    return {idx_};

  auto buff = parent_->indices();
  buff.push_back(idx());
  return buff;
}

Tensor* View::idx() {
  return idx_;
}

matcha::View ref(engine::View* internal) {
  return Engine::ref(internal);
}

engine::View* deref(const matcha::View& external) {
  return Engine::deref(external);
}

engine::View* deref(const matcha::View* external) {
  return Engine::deref(*external);
}

}