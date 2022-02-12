#include "bits_of_matcha/engine/tensor.h"

#include <matcha/engine>
#include <matcha/device>


namespace matcha {
namespace engine {


Tensor::Tensor(const Dtype& dtype, const Shape& shape)
  : dtype_{dtype}
  , shape_{shape}
  , buffer_{nullptr}
  , cpuBuffer_{nullptr}
  , in_{nullptr}
  , out_{createOut(dtype, shape)}
{
  setBuffer(device::Cpu().createBuffer(dtype, shape));
  status_ = {
    .data   = false,
    .update = false,
    .ready  = false
  };

  Debug() << "created Tensor " << this << " (from Dtype, Shape)";
}

Tensor::Tensor(Out* source)
  : dtype_{source->dtype()}
  , shape_{source->shape()}
  , in_{createIn(source)}
  , out_{createOut(source->dtype(), source->shape())}
  , buffer_{nullptr}
  , cpuBuffer_{nullptr}
{
  setBuffer(source->buffer());
  status_ = source->status();
  Debug() << "created Tensor " << this << " (from Out)";
}

In* Tensor::in() {
  return in_;
}

Out* Tensor::out() {
  return out_;
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

void Tensor::dataStatusChanged(In* in) {
  bool data = in->status().data;
  if (status_.data == data) return;
  status_.data = data;
  out_->dataStatusChanged();
}

void Tensor::updateStatusChanged(In* in) {
  if (status_.update) return;
  status_.update = true;
  out_->updateStatusChanged();
}

void Tensor::bufferChanged(In *in) {
  setBuffer(in->buffer());
}

void Tensor::eval(Out* target) {
  if (!status_.data) throw std::runtime_error("data not available yet");
  if (!status_.update) return;
  status_.update = false;

  Debug() << "eval Tensor " << this;

  if (in_ == nullptr) throw std::runtime_error("in is null");
  in_->eval();

  if (!status_.ready) {
    status_.ready = true;
  }
}

void Tensor::prune(Out* link) {
  if (referenced()) return;
  if (out_->linked()) return;

  if (in_ != nullptr) {
    delete in_;
  }

  delete this;
}

void Tensor::subst(Out* source) {
  if (source->dtype() != dtype() || source->shape() != shape()) {
    throw std::runtime_error("only tensors of matching form can be substituted");
  }
  if (in_ != nullptr) delete in_;
  in_ = createIn(source);
  setBuffer(in_->buffer());
  status_ = in_->status();

  out_->setBuffer(in_->buffer());
  out_->dataStatusChanged();
  out_->updateStatusChanged();
}

void Tensor::subst() {
  if (in_ != nullptr) {
    delete in_;
    in_ = nullptr;
  }
  setBuffer(nullptr);

  status_ = {
    .data   = false,
    .update = false,
    .ready  = false
  };

  out_->dataStatusChanged();
}

device::Buffer* Tensor::buffer() {
  return buffer_;
}

const device::Buffer* Tensor::buffer() const {
  return buffer_;
}

void Tensor::setBuffer(device::Buffer *buffer) {
  if (cpuBuffer_ != nullptr) {
    delete cpuBuffer_;
    cpuBuffer_ = nullptr;
  }
  buffer_ = buffer;
  out_->setBuffer(buffer_);
}

bool Tensor::hasData() const {
  if (buffer_ == nullptr) return false;
  if (buffer_->raw() == nullptr) return false;
  return true;
}

const std::byte* Tensor::getData() const {
  if (!status().data) throw std::runtime_error("data not available");
  if (cpuBuffer_ == nullptr) {
    cpuBuffer_ = device::Cpu().createBuffer(buffer());
    cpuBuffer_->prepare();
  }
  cpuBuffer_->update();
  return reinterpret_cast<std::byte*>(cpuBuffer_->raw());
}


}
}
