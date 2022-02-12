#include "bits_of_matcha/engine/input.h"

#include <matcha/engine>
#include <matcha/device>
#include <iostream>
#include <algorithm>


namespace matcha {
namespace engine {

Input::Input(const Dtype& dtype, const Shape& shape) {
  out_ = createOut(dtype, shape);
  buffer_ = device::Cpu().createBuffer(dtype, shape);
  buffer_->prepare();
  out()->setBuffer(buffer_);

  status_ = {
    .data = true,
    .update = false,
    .ready = true,
  };

  Debug() << "created Input " << this;
}

Input::Input(const Dtype& dtype, const Shape& shape, const std::vector<std::byte>& buffer)
{
  if (shape.size() * dtype.size() != buffer.size()) {
    throw std::invalid_argument("invalid buffer");
  }

  out_ = createOut(dtype, shape);
  buffer_ = device::Cpu().createBuffer(dtype, shape);
  buffer_->prepare();
  std::copy(
    std::begin(buffer), std::end(buffer),
    (std::byte*)(buffer_->raw())
  );

  out()->setBuffer(buffer_);

  status_ = {
    .data   = true,
    .update = false,
    .ready  = true,
  };

  Debug() << "created Input " << this;
}

Input::~Input() {
  Debug() << "deleted Input " << this;
}

Out* Input::out() {
  return out_;
}

const Dtype& Input::dtype() const {
  return buffer_->dtype();
}

const Shape& Input::shape() const {
  return buffer_->shape();
}

size_t Input::rank() const {
  return shape().rank();
}

size_t Input::size() const {
  return shape().size();
}

void Input::update(Tensor* tensor) {
  if (tensor->size() == 1 || tensor->shape() == shape()) {

    tensor->eval();
    const auto& source = tensor->buffer();
    buffer_->copy(source);

    if (tensor->rank() == 0) {
      // spread the scalar entry across the entire content
      float* buff = reinterpret_cast<float*>(buffer_->raw());
      std::fill(buff, buff + size(), buff[0]);
    }

  } else {
    std::cout << tensor->shape() << std::endl;
    std::cout << shape() << std::endl;
    throw std::invalid_argument("Input::update - shape mismatch");
  }
  out()->updateStatusChanged();
}

void Input::updateStatusChanged(In* in) {
  Debug() << "updated Input " << this;
  out()->updateStatusChanged();
}

void Input::prune(Out* link) {
  if (referenced()) return;
  if (out()->linked()) return;
  delete this;
}

template <class T>
T& Input::at(size_t position) {
  T* buffer = reinterpret_cast<T*>(buffer_->raw());
  return buffer[position];
}

template float& Input::at(size_t);

/*

const NodeLoader* Input::getLoader() const {
  return loader();
}

const NodeLoader* Input::loader() {
  static NodeLoader nl = {
    .type = "Input",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 0) throw std::invalid_argument("loading Input: invalid number of arguments");

      auto pos = is.tellg();

      auto lvals = FlowLoader::lvalues(is, ':');
      if (lvals.size() != 1 || lvals[0] != "dtype") {
        throw std::invalid_argument("semantic error: missing Input dtype");
      }
      auto dtype = FlowLoader::dtype(is);

      lvals = FlowLoader::lvalues(is, ':');
      if (lvals.size() != 1 || lvals[0] != "shape") {
        throw std::invalid_argument("semantic error: missing Input shape");
      }
      auto shape = FlowLoader::shape(is);

      lvals = FlowLoader::lvalues(is, ':');
      if (lvals.size() != 1 || lvals[0] != "data") {
        throw std::invalid_argument("semantic error: missing Input data");
      }
      auto floats = FlowLoader::flatFloats(is);
      auto bytes = reinterpret_cast<std::byte*>(&floats[0]);
      std::vector<std::byte> buffer(bytes, bytes + floats.size() * sizeof(float));
      return new Input(dtype, shape, buffer);
    }
  };
  return &nl;
}

void Input::save(std::ostream& os) const {
  os << "\n  ";
  FlowSaver::assignment(os, "dtype", ": ");
  FlowSaver::dtype(os, dtype());
  os << "\n  ";
  FlowSaver::assignment(os, "shape", ": ");
  FlowSaver::shape(os, shape());
  os << "\n  ";
  FlowSaver::assignment(os, "data", ": ");

  std::vector<float> data(size());
  FlowSaver::flatFloats(os, (const float*)(buffer_->raw()), shape(), 1);
}

*/

}
}
