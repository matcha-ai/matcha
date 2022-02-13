#include "bits_of_matcha/engine/params.h"

#include <matcha/engine>
#include <matcha/device>


namespace matcha {
namespace engine {


Params::Params(const Dtype& dtype, const Shape& shape)
  : dtype_{dtype}
  , shape_{shape}
{
  out_ = createOut(dtype, shape);
  buffer_ = device::Cpu().createBuffer(dtype, shape);
  buffer_->prepare();
  out()->setBuffer(buffer_);

  status_ = {
    .data   = true,
    .update = false,
    .ready  = true,
  };

  Debug() << "created Params";
}

Params::Params(const Dtype& dtype, const Shape& shape, const std::byte* data)
  : dtype_{dtype}
  , shape_{shape}
{
  out_ = createOut(dtype, shape);
  buffer_ = device::Cpu().createBuffer(dtype, shape);
  buffer_->prepare();

  std::byte* buff = reinterpret_cast<std::byte*>(buffer_->raw());
  std::copy(data, data + size(), buff);

  out()->setBuffer(buffer_);

  status_ = {
    .data   = true,
    .update = false,
    .ready  = true,
  };

  Debug() << "created Params";
}

Params::Params(const Dtype& dtype, const Shape& shape, Tensor* init)
  : Params(dtype, shape)
{
  update(init);
}

Params::Params(const Dtype& dtype, const Shape& shape, Stream* init)
  : Params(dtype, shape)
{
  Tensor* temp = new Tensor(dtype, shape);
  init->open(-1, temp);
  update(temp);
  temp->prune();

  Debug() << "created Params";
}

Params::Params(Tensor* tensor)
  : dtype_{tensor->dtype()}
  , shape_{tensor->shape()}
{
  auto& dtype = tensor->dtype();
  auto& shape = tensor->shape();

  buffer_ = device::Cpu().createBuffer(dtype, shape);
  buffer_->prepare();
  out_ = createOut(dtype, shape);
  out()->setBuffer(buffer_);

  Debug() << "created Params";

  update(tensor);
}

Params::Params(Input* init)
  : Params(init->dtype(), init->shape())
{}

Params::Params(Stream* init)
  : dtype_{Dtype::Float}
  , shape_{}
{
  std::cout << "TODO!!!" << std::endl;
}

const Dtype& Params::dtype() const {
  return dtype_;
}

const Shape& Params::shape() const {
  return shape_;
}

size_t Params::rank() const {
  return shape().rank();
}

size_t Params::size() const {
  return shape().size();
}

void Params::update(Tensor* tensor) {
  tensor->eval();
  auto* source = tensor->buffer();

  if (tensor->shape() == shape()) {

    buffer_->copy(source);

  } else if (tensor->size() == 1) {

    auto* temp = device::Cpu().createBuffer(source);
    temp->prepare();
    temp->update();
    float val = *(const float*)temp->raw();

    float* buff = reinterpret_cast<float*>(buffer_->raw());
    std::fill(buff, buff + size(), val);

    delete temp;

  } else {
    throw std::invalid_argument("Params::update - shape mismatch");
  }
  Debug() << "updated Params " << this;
  out()->updateStatusChanged();
}

Out* Params::out() {
  return out_;
}

void Params::prune(Out* link) {
  if (referenced()) return;
  if (out()->linked()) return;
  delete this;
}

/*
const NodeLoader* Params::getLoader() const {
  return loader();
}

const NodeLoader* Params::loader() {
  static NodeLoader nl = {
    .type = "Params",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 0) throw std::invalid_argument("loading Params: invalid number of arguments");

      auto pos = is.tellg();

      auto lvals = FlowLoader::lvalues(is, ':');
      if (lvals.size() != 1 || lvals[0] != "dtype") {
        throw std::invalid_argument("semantic error: missing Params dtype");
      }
      auto dtype = FlowLoader::dtype(is);

      lvals = FlowLoader::lvalues(is, ':');
      if (lvals.size() != 1 || lvals[0] != "shape") {
        throw std::invalid_argument("semantic error: missing Params shape");
      }
      auto shape = FlowLoader::shape(is);

      lvals = FlowLoader::lvalues(is, ':');
      if (lvals.size() != 1 || lvals[0] != "data") {
        throw std::invalid_argument("semantic error: missing Params data");
      }
      auto floats = FlowLoader::flatFloats(is);
      auto bytes = reinterpret_cast<std::byte*>(&floats[0]);

      return new Params(dtype, shape, bytes);
    }
  };
  return &nl;
}

void Params::save(std::ostream& os) const {
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
