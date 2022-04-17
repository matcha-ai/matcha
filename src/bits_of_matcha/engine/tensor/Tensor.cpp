#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/tensor/TensorCtx.h"
#include "bits_of_matcha/engine/tensor/iterations.h"
#include "bits_of_matcha/engine/memory/memory.h"
#include "bits_of_matcha/engine/flow/Tracer.h"


namespace matcha::engine {

Tensor::Tensor(const Frame& frame, Op* op)
  : frame_(frame)
  , ctx_(this)
  , buffer_(nullptr)
  , op_(op)
{
  Tracer::handleNewTensor(this);
}

Tensor::Tensor(const Dtype& dtype, const Shape& shape, Op* op)
  : Tensor(Frame{dtype, shape}, op)
{}

Tensor::~Tensor() {
  if (buffer_) {
    buffer_->unbind();
  }
}

const Frame& Tensor::frame() const {
  return frame_;
}

const Dtype& Tensor::dtype() const {
  return *frame_.dtype();
}

const Shape& Tensor::shape() const {
  return *frame_.shape();
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

Buffer* Tensor::buffer() {
  return buffer_;
}

Buffer* Tensor::malloc() {
  if (buffer_) return buffer_;
  buffer_ = engine::malloc(bytes());
  return buffer_;
}

TensorCtx& Tensor::ctx() {
  return ctx_;
}

Op* Tensor::op() {
  return op_;
}

void Tensor::setOp(Op* op) {
  op_ = op;
}

void Tensor::repr(std::ostream& os) {
  if (!buffer()) {
    os << dtype() << shape() << std::endl;
    return;
  }

  auto floats = buffer()->as<float*>();

  if (rank() == 0) {
    os << "tensor {" << floats[0] << "}" << std::endl;
    return;
  }

  auto iter = MatrixStackIteration(shape());
  bool oneline = iter.rows == 1;
  auto indent = std::string(2, ' ');

  os << "tensor {";
  if (!oneline) os << "\n";
  for (int matrix = 0; matrix < iter.amount; matrix++) {
    if (matrix != 0) os << "\n";
    for (int row = 0; row < iter.rows; row++) {
      if (row != 0) os << "\n";
      os << indent;

      for (int col = 0; col < iter.cols; col++) {
        os << floats[matrix * iter.amount + row * iter.cols + col] << " ";
      }
    }
  }
  if (!oneline) os << "\n";
  os << "}";
}

tensor ref(Tensor* internal) {
  static_assert(sizeof(tensor) == sizeof(void*));
  auto p = (tensor*) &internal;
  tensor t = *p;
  return t;
}

Tensor* unref(tensor& external) {
  return unref(&external);
}

Tensor* unref(tensor* external) {
  static_assert(sizeof(tensor) == sizeof(void*));
  void** pp = reinterpret_cast<void**>(external);
  void*& p = *pp;
  auto internal = (Tensor*)p;
  p = nullptr;
  return internal;
}

Tensor* deref(const tensor* external) {
  static_assert(sizeof(tensor) == sizeof(void*));
  const tensor** tensorpp = &external;
  auto internalppp = (Tensor***)tensorpp;
  return **internalppp;
}

Tensor* deref(const tensor& external) {
  return deref(&external);
}

Tensor* full(float value, const Shape& shape) {
  auto tensor = new Tensor(Float, shape);

  auto buffer = tensor->malloc();
  auto floats = buffer->as<float*>();
  std::fill(floats, floats + tensor->size(), value);

  return tensor;
}

Tensor* zeros(const Shape& shape) {
  return full(0, shape);
}

Tensor* ones(const Shape& shape) {
  return full(1, shape);
}

Tensor* eye(const Shape& shape) {
  auto tensor = zeros(shape);
  auto floats = tensor->buffer()->as<float*>();

  auto iter = MatrixStackIteration(shape);
  for (int matrix = 0; matrix < iter.amount; matrix++) {
    for (int row = 0; row < std::min(iter.rows, iter.cols); row++) {
      floats[matrix * iter.size + row * (iter.cols + 1)] = 1;
    }
  }

  return tensor;
}

}