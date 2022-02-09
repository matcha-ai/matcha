#include "bits_of_matcha/engine/input.h"
#include "bits_of_matcha/engine/stream.h"
#include "bits_of_matcha/engine/nodeloader.h"
#include "bits_of_matcha/engine/flowloader.h"
#include "bits_of_matcha/engine/flowsaver.h"

#include <matcha/device>
#include <iostream>
#include <algorithm>


namespace matcha {
namespace engine {

Input::Input(const Dtype& dtype, const Shape& shape)
  : Node{}
{
  addOut(dtype, shape);
  buffer_ = device::Cpu().createBuffer(dtype, shape);
  buffer_->prepare();
  out(0)->setBuffer(buffer_);
}

Input::Input(const Dtype& dtype, const Shape& shape, const std::vector<std::byte>& buffer)
  : Node{}
{
  if (shape.size() * dtype.size() != buffer.size()) {
    throw std::invalid_argument("invalid buffer");
  }

  addOut(dtype, shape);
  buffer_ = device::Cpu().createBuffer(dtype, shape);
  buffer_->prepare();
  std::copy(
    std::begin(buffer), std::end(buffer),
    (std::byte*)(buffer_->raw())
  );

  out(0)->setBuffer(buffer_);
}

Input::Input(const Stream& stream)
  : Input(stream.generateNext())
{}

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

void Input::eval(Tensor* target) {

}

void Input::require() {

}

template <class T>
T& Input::at(size_t position) {
  T* buffer = reinterpret_cast<T*>(buffer_->raw());
  return buffer[position];
}

template float& Input::at(size_t);

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

}
}
