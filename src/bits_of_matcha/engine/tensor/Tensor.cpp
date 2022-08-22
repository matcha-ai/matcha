#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/tensor/iterations.h"
#include "bits_of_matcha/engine/memory/memory.h"
#include "bits_of_matcha/engine/lambda/Tracer.h"
#include "bits_of_matcha/Engine.h"
#include "bits_of_matcha/print.h"

#include <sstream>


namespace matcha::engine {

Tensor::Tensor(const Frame& frame)
  : frame_(frame)
  , op_(nullptr)
{
//  print("created tensor ", this);
//  Tracer::handleNewTensor(this);
}

Tensor::Tensor(const Dtype& dtype, const Shape& shape)
  : Tensor(Frame(dtype, shape))
{}

Tensor::~Tensor() {
//  print("destroying ", this, " (", frame_.string(), ")");
//  free();
}

const Frame& Tensor::frame() const {
  return frame_;
}

const Dtype& Tensor::dtype() const {
  return frame_.dtype();
}

const Shape& Tensor::shape() const {
  return frame_.shape();
}

size_t Tensor::rank() const {
  return shape().rank();
}

size_t Tensor::size() const {
  return shape().size();
}

size_t Tensor::bytes() const {
  return dtype().size() * size();
}

Buffer& Tensor::buffer() {
  return buffer_;
}

Buffer& Tensor::malloc() {
  buffer_.malloc(bytes());
  return buffer_;
}

Buffer& Tensor::share(Buffer& buffer) {
  buffer_ = buffer;
  return buffer_;
}

Buffer& Tensor::share(Tensor* tensor) {
  if (!tensor) throw std::runtime_error("can't share buffer: tensor is null");
  buffer_ = tensor->buffer_;
  return buffer_;
}

Buffer& Tensor::free() {
  buffer_.free();
  return buffer_;
}


Op* Tensor::op() const {
  return op_;
}

void Tensor::setOp(Op* op) {
  op_ = op;
}

void* Tensor::readData() {
  if (tracing())
    throw std::runtime_error("reading tensor data directly inside the Flow is forbidden");

  if (!buffer_) return nullptr;
  return buffer_.payload();
}

tensor ref(Tensor* internal) {
  return Engine::ref(internal);
}

Tensor* unref(tensor& external) {
  return Engine::unref(external);
}

Tensor* unref(tensor* external) {
  return Engine::unref(external);
}

Tensor* deref(const tensor* external) {
  return Engine::deref(external);
}

Tensor* deref(const tensor& external) {
  return Engine::deref(external);
}

std::vector<tensor> ref(const std::vector<Tensor*>& internals) {
  std::vector<tensor> result;
  result.reserve(internals.size());
  for (auto&& internal: internals)
    result.push_back(ref(internal));
  return result;
}

std::vector<Tensor*> deref(const std::vector<tensor>& externals) {
  std::vector<Tensor*> result;
  result.reserve(externals.size());
  for (auto&& external: externals)
    result.push_back(deref(external));
  return result;
}

std::vector<Tensor*> deref(const std::vector<tensor*>& externals) {
  std::vector<Tensor*> result;
  result.reserve(externals.size());
  for (auto&& external: externals)
    result.push_back(deref(external));
  return result;
}

std::vector<Tensor*> unref(std::vector<tensor>& externals) {
  std::vector<Tensor*> result;
  result.reserve(externals.size());
  for (auto&& external: externals)
    result.push_back(unref(external));
  return result;
}

std::vector<Tensor*> unref(std::vector<tensor*>& externals) {
  std::vector<Tensor*> result;
  result.reserve(externals.size());
  for (auto&& external: externals)
    result.push_back(unref(external));
  return result;
}

}