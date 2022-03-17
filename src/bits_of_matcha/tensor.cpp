#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/Slice.h"
#include "bits_of_matcha/print.h"
#include "bits_of_matcha/engine/Tensor.h"
#include "bits_of_matcha/engine/flow/FlowFunctionContext.h"
#include "bits_of_matcha/engine/cpu/buffer.h"
#include "bits_of_matcha/engine/fn.h"

#include "bits_of_matcha/fn/identity.h"
#include "bits_of_matcha/fn/basic_arithmetic.h"
#include "bits_of_matcha/fn/dot.h"
#include "bits_of_matcha/fn/exponents.h"
#include "bits_of_matcha/fn/transpose.h"
#include "bits_of_matcha/fn/reshape.h"


namespace matcha {

tensor::tensor()
  : internal_{nullptr}
{
  bind(new engine::Tensor());
}

tensor::tensor(const Dtype& dtype, const Shape& shape)
  : internal_{nullptr}
{
  bind(new engine::Tensor(dtype, shape));
}

tensor::tensor(const tensor& t)
  : internal_{nullptr}
{
  tensor id = fn::identity(t);
  internal_ = id.internal_;
  id.internal_ = nullptr;
}

tensor::tensor(float content)
  : internal_{nullptr}
{
  bind(new engine::Tensor(Float, {}));
  auto buffer = new engine::cpu::Buffer(sizeof(float));
  auto& val = *(float*) buffer->payload();
  val = content;
  internal_->shareBuffer(buffer);
}

tensor::tensor(std::initializer_list<float> content)
  : internal_{nullptr}
{
  bind(new engine::Tensor(Float, {(unsigned) content.size()}));
  auto data = new float[content.size()];
  std::copy(std::begin(content), std::end(content), data);
  auto buffer = new engine::cpu::Buffer(sizeof(float) * content.size(), data);
  internal_->shareBuffer(buffer);
}

tensor::tensor(std::initializer_list<std::initializer_list<float>> content)
  : internal_{nullptr}
{
  unsigned dims[] = {0, 0};
  dims[0] = content.size();
  for (auto& row: content) {
    if (dims[1] == 0) {
      if (row.size() == 0) throw std::invalid_argument("empty axis");
      dims[1] = row.size();
    } else if (row.size() != dims[1]) {
      throw std::invalid_argument("irregular axes");
    }
  }

  size_t size = dims[0] * dims[1];
  auto memory = new float[size];

  int i = 0;
  for (auto& row: content) {
    const float* rowdata = std::data(row);
    std::copy(
      std::begin(row), std::end(row),
      memory + i * dims[1]
    );
    i++;

    bind(new engine::Tensor(Float, {(unsigned) content.size()}));
    auto data = std::data(content);
    auto buffer = new engine::cpu::Buffer(sizeof(float) * content.size(), (void*) data);
    internal_->shareBuffer(buffer);
  }


}

tensor::tensor(std::initializer_list<std::initializer_list<std::initializer_list<float>>> content) {

}

tensor::~tensor() {
  bind(nullptr);
}

tensor& tensor::operator=(const tensor& t) {
  t.assertNotQuery();
  assertNotQuery();

//  print("operator=(const tensor&)");
  tensor id = fn::identity(t);

  internal_->unref();
  internal_ = id.internal_;
  id.internal_ = nullptr;

  return *this;
}

Slice tensor::operator[](const Shape::Range& range) {
  return Slice(this, range);
}

bool tensor::frame() const {
  assertNotQuery();
  return !internal_->frame()->null();
}

const Dtype& tensor::dtype() const {
  assertNotQuery();
  return internal_->dtype();
}

const Shape& tensor::shape() const {
  assertNotQuery();
  return internal_->shape();
}

size_t tensor::size() const {
  return internal_->size();
}

size_t tensor::rank() const {
  return internal_->rank();
}

tensor tensor::transpose() const {
  return fn::transpose(*this);
}

tensor tensor::t() const {
  return transpose();
}

tensor tensor::reshape(const Shape::Reshape& shape) const {
  return fn::reshape(*this, shape);
}

tensor tensor::map(const UnaryFn& fn) const {
  return fn(*this);
}

tensor tensor::map(const tensor& linear) const {
  return linear.dot(*this);
}

tensor tensor::map(const tensor& linear, const tensor& affine) const {
  return linear.dot(*this) + affine;
}

tensor tensor::dot(const tensor& tensor) const {
  return fn::dot(*this, tensor);
}

tensor tensor::pow(const tensor& exponent) const {
  return fn::pow(*this, exponent);
}

tensor tensor::nrt(const tensor& exponent) const {
  return fn::nrt(*this, exponent);
}

tensor tensor::norm() const {
  return 1;
}

tensor tensor::normalize() const {
  return fn::divide(*this, norm());
}

void* tensor::data() {
  assertNotQuery();
  if (internal_->flow_) {
    throw std::runtime_error("access to tensor buffers inside a flow is forbidden; you have to use a built-in matcha function");
  }

  internal_->readData();
  return internal_->data();
}

tensor::tensor(engine::Tensor* internal)
  : internal_{nullptr}
{
  bind(internal);
  if (internal->eager()) {
    internal->compute();
  }
}

void tensor::bind(engine::Tensor* tensor) {
  if (internal_) internal_->unref();
  internal_ = tensor;
  if (internal_) internal_->ref();
}

engine::Flow* tensor::flowQuery(const UnaryFn& fn) {
  tensor queryTensor(nullptr);
//  try {
//    tensor responseTensor = fn(queryTensor);
//  } catch (engine::FlowQueryResponse& r) {
//    return r.flow;
//  }
  return nullptr;
}

bool tensor::getFlowQuery() const {
  return internal_ == nullptr;
}

void tensor::assertNotQuery() const {
//  if (getFlowQuery()) throw engine::FlowQueryResponse();
}

tensor tensor::full(const Shape& shape, float value) {
  auto t = new engine::Tensor(Float, shape);
  auto b = t->writeBuffer();

  auto vals = (float*) b->payload();
  std::fill(vals, vals + shape.size(), value);

  return tensor(t);
}


tensor tensor::zeros(const Shape& shape) {
  return full(shape, 0);
}

tensor zeros(const Shape& shape) {
  return tensor::zeros(shape);
}

tensor tensor::ones(const Shape& shape) {
  return full(shape, 1);
}

tensor ones(const Shape& shape) {
  return tensor::ones(shape);
}

tensor tensor::eye(const Shape& shape) {
  auto t = new engine::Tensor(Float, shape);
  auto b = t->writeBuffer();

  auto vals = (float*) b->payload();
  if (shape.rank() == 0) {
    vals[0] = 1;
  } else {
    std::fill(vals, vals + shape.size(), 0);

    auto iter = engine::fn::MatrixStackIteration(shape);
    for (int matrix = 0; matrix < iter.amount; matrix++) {
      for (int row = 0; row < std::min(iter.rows, iter.cols); row++) {
        vals[matrix * iter.size + row * (iter.cols + 1)] = 1;
      }
    }
  }

  return tensor(t);
}

tensor full(const Shape& shape, float value) {
  return tensor::full(shape, value);
}

tensor eye(const Shape& shape) {
  return tensor::eye(shape);
}

}

std::ostream& operator<<(std::ostream& os, const matcha::tensor& tensor) {
  tensor.assertNotQuery();
  if (!tensor.frame()) {
    os << "tensor {}" << std::endl;
    return os;
  }

  auto t = tensor.internal_;
  t->readData();
  void* data = t->data();
  if (!data) {
    os << tensor.dtype() << tensor.shape() << std::endl;
    return os;
  }

  auto iter = matcha::engine::fn::MatrixStackIteration(
    tensor.rank() ? tensor.shape() : matcha::Shape{1, 1}
  );

  bool block;
  switch (tensor.rank()) {
    case 0:
      block = false;
      break;
    case 1:
      block = tensor.size() > 1;
      break;
    case 2:
      block = tensor.shape()[0] > 1;
      break;
    default:
      block = true;
  }

  size_t size = iter.rows * iter.cols;

  os << "tensor {";
  if (block) os << "\n";

  for (int matrix = 0; matrix < iter.amount; matrix++) {
    if (matrix != 0) os << "\n\n";
    for (int row = 0; row < iter.rows; row++) {
      if (row != 0) os << "\n";
      for (int col = 0; col < iter.cols; col++) {
        if (col == 0) {
          if (block) os << "  ";
        } else {
          os << " ";
        }

        switch (tensor.dtype()) {
          case matcha::Float:
            os << ((float*) data)[matrix * size + row * iter.cols + col];
            break;
          default:
            os << "?";
        }
      }
    }
  }

  if (block) os << "\n";
  os << "}" << std::endl;
  return os;
}

