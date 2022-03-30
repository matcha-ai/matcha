#include "bits_of_matcha/engine/Tensor.h"
#include "bits_of_matcha/engine/Node.h"
#include "bits_of_matcha/engine/memory.h"
#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/print.h"

#include <execution>


namespace matcha::engine {


Tensor::Tensor(Frame frame)
  : frame_{std::move(frame)}
  , bufferInternal_{nullptr}
  , bufferExternal_{nullptr}
  , source_{nullptr}
  , refs_{0}
  , reqs_{0}
  , mode_{Untraced}
  , ctxId_{-1}
  , flow_{false}
{
  auto tracer = Tracer::current();
  if (tracer){
    tracer->add(this);
  }
}

Tensor::Tensor(const Dtype& dtype, const Shape& shape)
  : Tensor(Frame(dtype, shape))
{}

Tensor::Tensor()
  : Tensor(Frame())
{}

Tensor::~Tensor() {
  delete source_;
  if (bufferInternal_) bufferInternal_->unbind();
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
  if (!bufferInternal_) return;

  if (!bufferExternal_) {
    if (bufferInternal_->uses(CPU)) {
      bufferExternal_ = bufferInternal_;
    } else {
      bufferExternal_ = malloc(bytes());
    }
  }

  if (!bufferInternal_->uses(CPU)) {
    Buffer::transfer(bufferInternal_, bufferExternal_);
  }
}

void* Tensor::data() {
  if (!bufferExternal_) return nullptr;
  return bufferExternal_->payload();
}

Buffer* Tensor::buffer() {
  return bufferInternal_;
}

void Tensor::shareBuffer(Buffer* buffer) {
  if (bufferInternal_) bufferInternal_->unbind();
  bufferInternal_ = buffer;
  if (bufferInternal_) bufferInternal_->bind();
}

void Tensor::shareBuffer(Tensor* tensor) {
  shareBuffer(tensor->bufferInternal_);
}

void Tensor::stealBuffer(Tensor* tensor) {
  if (bufferInternal_) bufferInternal_->unbind();
  bufferInternal_ = tensor->bufferInternal_;
  tensor->bufferInternal_ = nullptr;
}

Buffer* Tensor::writeBuffer(const Device::Concrete& device) {
  if (device.hosts(bufferInternal_) && !bufferInternal_->shared()) return bufferInternal_;
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
  if (lazy()) {
    throw std::runtime_error("can only compute eager tensors");

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
  if (bufferInternal_) return bufferInternal_->device();
  throw std::runtime_error("tensor without source and device");
}

bool Tensor::uses(const Device::Concrete& dev) const {
  return *device() == dev;
}

bool Tensor::uses(const Device::Concrete* dev) const {
  return *device() == *dev;
}

int Tensor::ctxId() const {
  return ctxId_;
}

void Tensor::setCtxId(int ctxId) {
  ctxId_ = ctxId;
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

bool Tensor::eager() const {
  return source_ && !source_->flow();
}

bool Tensor::lazy() const {
  return (!source_ || source_->flow());
}

unsigned Tensor::mode() const {
  return mode_;
}

void Tensor::setMode(unsigned mode) {
  mode_ = mode;
}

void Tensor::assign(Tensor* source) {
  frame_ = source->frame_;
}

void Tensor::update(Tensor* source) {
  if (source->frame_ != frame_) throw std::invalid_argument("shape mismatch");
}

Tensor* deref(const matcha::tensor& tensor) {
  return deref(&tensor);
}

Tensor* deref(const matcha::tensor* tensor) {
  return tensor->internal_;
}

Tensor* Tensor::full(float value, const Shape& shape) {
  auto t = new Tensor(Float, shape);
  auto b = t->writeBuffer();

  auto vals = (float*) b->payload();
  std::fill(std::execution::par_unseq, vals, vals + shape.size(), value);

  return t;
}


}