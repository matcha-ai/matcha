#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/engine/flowBuilder.h"
#include "bits_of_matcha/engine/pipeBuffers.h"
#include "bits_of_matcha/engine/node.h"
#include "bits_of_matcha/engine/memory.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {


Tensor::Tensor(Frame frame)
  : frame_{std::move(frame)}
  , buffer_{nullptr}
  , cpuBuffer_{nullptr}
  , source_{nullptr}
  , refs_{0}
  , reqs_{0}
{
  auto build = FlowBuilder::current();
  flow_ = build;
  if (build) build->add(this);
}

Tensor::Tensor(const Dtype& dtype, const Shape& shape)
  : Tensor(Frame(dtype, shape))
{}

Tensor::Tensor()
  : Tensor(Frame())
{}

Tensor::~Tensor() {
  delete source_;
  if (buffer_) buffer_->unbind();
}

const Frame* Tensor::frame() const {
  return &frame_;
}

const Dtype& Tensor::dtype() const {
  return *frame_.dtype();
}

const Shape& Tensor::shape() const {
  return *frame_.shape();
}

size_t Tensor::size() const {
  return shape().size();
}

size_t Tensor::rank() const {
  return shape().rank();
}

size_t Tensor::bytes() const {
  return frame_.bytes();
}

void Tensor::readData() {
  if (source_) {
    compute();
  }
  if (!buffer_) return;

  if (!cpuBuffer_) {
    if (buffer_->uses(CPU)) {
      cpuBuffer_ = buffer_;
    } else {
      cpuBuffer_ = malloc(bytes());
    }
  }

  if (!buffer_->uses(CPU)) {
    Buffer::transfer(buffer_, cpuBuffer_);
  }
}

void* Tensor::data() {
  if (!cpuBuffer_) return nullptr;
  return cpuBuffer_->payload();
}

Buffer* Tensor::buffer() {
  return buffer_;
}

void Tensor::shareBuffer(Buffer* buffer) {
  if (buffer_) buffer_->unbind();
  buffer_ = buffer;
  if (buffer_) buffer_->bind();
}

void Tensor::shareBuffer(Tensor* tensor) {
  shareBuffer(tensor->buffer_);
}

void Tensor::stealBuffer(Tensor* tensor) {
  if (buffer_) buffer_->unbind();
  buffer_ = tensor->buffer_;
  tensor->buffer_ = nullptr;
}

Buffer* Tensor::writeBuffer(const Device::Concrete& device) {
  if (device.hosts(buffer_) && !buffer_->shared()) return buffer_;
  auto buffer = malloc(bytes(), device);
  shareBuffer(buffer);
  return buffer;
}

void Tensor::setSource(Node* source) {
  source_ = source;
}

Node* Tensor::source() {
  return source_;
}

void Tensor::compute() {
  if (flow_) {

  } else {
    if (!source_) return;
    source_->init();
    source_->run();
    delete source_;
    source_ = nullptr;
  }
}

const Device::Concrete* Tensor::device() const {
  if (source_) return source_->device();
  if (buffer_) return buffer_->device();
  throw std::runtime_error("tensor without source and device");
}

bool Tensor::uses(const Device::Concrete& dev) const {
  return *device() == dev;
}

bool Tensor::uses(const Device::Concrete* dev) const {
  return *device() == *dev;
}

void Tensor::ref() {
  refs_++;
}

void Tensor::unref() {
  if (refs_ == 0) throw std::runtime_error("ref count is already 0");
  refs_--;
  if (!refs_ && !reqs_) delete this;
}

void Tensor::req() {
  reqs_++;
}

void Tensor::unreq() {
  if (reqs_ == 0) throw std::runtime_error("req count is already 0");
  reqs_--;
  if (!refs_ && !reqs_) delete this;
}

unsigned Tensor::refs() const {
  return refs_;
}

unsigned Tensor::reqs() const {
  return reqs_;
}

bool Tensor::flow() const {
  return flow_;
}

Tensor* deref(const matcha::Tensor& tensor) {
  return deref(&tensor);
}

Tensor* deref(const matcha::Tensor* tensor) {
  return tensor->pimpl_;
}


}