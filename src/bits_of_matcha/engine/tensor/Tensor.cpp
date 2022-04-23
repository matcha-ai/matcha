#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/tensor/TensorCtx.h"
#include "bits_of_matcha/engine/tensor/iterations.h"
#include "bits_of_matcha/engine/memory/memory.h"
#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/print.h"

#include <sstream>


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
//  print("destroying ", this, " (", frame_.string(), ")");
  free();
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

void Tensor::shareBuffer(Buffer* buffer) {
  if (buffer == buffer_) return;
  if (buffer_) buffer_->unbind();
  buffer_ = buffer;
  if (buffer_) buffer->bind();
}

void Tensor::shareBuffer(Tensor* tensor) {
  shareBuffer(tensor->buffer());
}

void Tensor::free() {
  if (buffer_) buffer_->unbind();
  buffer_ = nullptr;
}

void Tensor::repr(std::ostream& os) {
//  os << this << " tensorrrr" << std::endl;
//  print(frame_.string());
//  print(buffer());
//  os << this << " ";
  if (frame().null()) {
    os << "NullTensor";
    return;
  }

  if (!buffer()) {
    os << dtype() << shape();
    return;
  }

//  print("buffer is now: ", buffer());
  auto floats = buffer()->as<float*>();

  if (rank() == 0) {
    os << floats[0];
    return;
  }

  auto iter = MatrixStackIteration(shape());
  bool oneline = iter.rows == 1;

  size_t cellW = 0;
  for (size_t i = 0; i < size(); i++) {
    std::stringstream ss;
    ss << floats[i];
    size_t w = ss.str().size();
    cellW = std::max(cellW, w);
  }

  size_t termCols = 80;
  int skipSize = (int) iter.cols - (int)(termCols / cellW);
  int skipBegin = ((int) iter.cols - skipSize) / 2;

  int indent = 0;
//  os << "[";

  for (int matrix = 0; matrix < iter.amount; matrix++) {

    if (iter.rows > 1) {
      if (matrix != 0) {
        os << "],\n";
      }
      os << "[";
      indent++;
    }

    for (int row = 0; row < iter.rows; row++) {
      if (row == 0) {
        os << "[";
      } else {
        os << "]\n";
        os << std::string(indent, ' ') << "[";
      }

      for (int col = 0; col < iter.cols; col++) {
        if (col != 0) os << " ";
        std::stringstream ss;
        float val = floats[matrix * iter.amount + row * iter.cols + col];
        ss << val;
        std::string temp = ss.str();
        os << temp << std::string(cellW - temp.size(), ' ');
      }
    }
    os << "]";

    if (rank() >= 2) {
      indent--;
    }
  }

  if (iter.rows > 1) {
    os << "]";
  }

}

tensor ref(Tensor* internal) {
  static_assert(sizeof(tensor) == sizeof(void*));
  return tensor(internal);
}

Tensor* unref(tensor& external) {
  return unref(&external);
}

Tensor* unref(tensor* external) {
  static_assert(sizeof(tensor) == sizeof(void*));
  void** pp = reinterpret_cast<void**>(external);
  void*& p = *pp;
  auto internal = (Tensor*) p;
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

}