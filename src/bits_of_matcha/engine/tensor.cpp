#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/engine/params.h"
#include "bits_of_matcha/engine/stream.h"
#include "bits_of_matcha/engine/input.h"
#include "bits_of_matcha/tensor.h"

#include <matcha/device>


namespace matcha {
namespace engine {


Tensor::Tensor(const Dtype& dtype, const Shape& shape)
  : dtype_{dtype}
  , shape_{shape}
  , in_{nullptr}
  , buffer_{nullptr}
  , cpuBuffer_{nullptr}
  , required_{false}
  , ready_{false}
{
  setBuffer(device::Cpu().createBuffer(dtype, shape));
}

Tensor::Tensor(Node* in, device::Buffer* buffer)
  : dtype_{buffer->dtype()}
  , shape_{buffer->shape()}
  , in_{in}
  , buffer_{nullptr}
  , cpuBuffer_{nullptr}
  , required_{false}
  , ready_{in->ready()}
{
  setBuffer(buffer);
}

Tensor::Tensor(device::Buffer* buffer)
  : dtype_{buffer->dtype()}
  , shape_{buffer->shape()}
  , in_{nullptr}
  , buffer_{nullptr}
  , cpuBuffer_{nullptr}
  , required_{false}
  , ready_{false}
{
  setBuffer(buffer);
}

const Dtype& Tensor::dtype() const {
  return dtype_;
}

const Shape& Tensor::shape() const {
  return shape_;
}

size_t Tensor::rank() const {
  return shape().rank();
}

size_t Tensor::size() const {
  return shape().size();
}

void Tensor::require() {
  if (required()) return;
  required_ = true;
  for (auto* out: outs_) out->require();
}

bool Tensor::required() const {
  return required_;
}

void Tensor::unrequire() const {
  required_ = false;
}

void Tensor::eval() {
  if (!ready()) return;
  if (!required()) return;
  unrequire();
  if (in_ != nullptr) in_->eval(this);
}

device::Buffer* Tensor::buffer() {
  return buffer_;
}

const device::Buffer* Tensor::buffer() const {
  return buffer_;
}

void Tensor::setBuffer(device::Buffer *buffer) {
  if (buffer_ != nullptr) {
    delete buffer_;
    delete cpuBuffer_;
  }
  buffer_ = buffer;
  cpuBuffer_ = device::Cpu().createBuffer(buffer);
  cpuBuffer_->prepare();
  require();
}

bool Tensor::ready() const {
  return ready_;
}

void Tensor::setReady(bool ready) const {
  if (ready_ == ready) return;
  ready_ = ready;
}

bool Tensor::hasData() const {
  if (buffer_ == nullptr) return false;
  if (buffer_->raw() == nullptr) return false;
  return true;
}

const std::byte* Tensor::getData() const {
  cpuBuffer_->update();
  return reinterpret_cast<std::byte*>(cpuBuffer_->raw());
}

void Tensor::considerPruning() {
  if (referenced()) return;
  if (!outs_.empty()) return;

  if (in_ == nullptr) {
    delete this;
    return;
  }

  if (in_->closeOut(this)) {
    delete this;
  }
  in_->considerPruning();
}

void Tensor::bindOut(Node* out) {
  outs_.insert(out);
}

void Tensor::unbindOut(Node* out) {
  outs_.erase(out);
}

void Tensor::bindIn(Node* in, unsigned edgeId) {
  if (in_ != nullptr) throw std::runtime_error("Tensor: in is already bound");
  in_ = in;
  edgeId_ = edgeId;
  setReady(in->ready());
}

void Tensor::unbindIn(Node* in) {
  if (in_ != in) throw std::runtime_error("Tensor: in is not bound");
  in_ = nullptr;
  setReady(false);
}

unsigned Tensor::edgeId() const {
  return edgeId_;
}

void Tensor::setEdgeId(unsigned int edgeId) const {
  edgeId_ = edgeId;
}


}
}
