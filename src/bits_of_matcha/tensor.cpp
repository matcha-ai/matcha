#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/print.h"
#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/engine/flowContext.h"
#include "bits_of_matcha/engine/cpu/buffer.h"
#include "bits_of_matcha/engine/fn.h"

#include "bits_of_matcha/fn/identity.h"
#include "bits_of_matcha/fn/basicArithmetic.h"
#include "bits_of_matcha/fn/dot.h"
#include "bits_of_matcha/fn/exponents.h"
#include "bits_of_matcha/fn/transpose.h"
#include "bits_of_matcha/fn/reshape.h"


namespace matcha {

Tensor::Tensor()
  : pimpl_{nullptr}
{
  bind(new engine::Tensor());
}

Tensor::Tensor(const Dtype& dtype, const Shape& shape)
  : pimpl_{nullptr}
{
  bind(new engine::Tensor(dtype, shape));
}

Tensor::Tensor(const Tensor& tensor)
  : pimpl_{nullptr}
{
  Tensor id = fn::identity(tensor);
  pimpl_ = id.pimpl_;
  id.pimpl_ = nullptr;
}

Tensor::Tensor(float content)
  : pimpl_{nullptr}
{
  bind(new engine::Tensor(Float, {}));
  auto buffer = new engine::cpu::Buffer(sizeof(float));
  auto& val = *(float*) buffer->payload();
  val = content;
  pimpl_->shareBuffer(buffer);
}

Tensor::Tensor(std::initializer_list<float> content)
  : pimpl_{nullptr}
{
  bind(new engine::Tensor(Float, {(unsigned) content.size()}));
  auto data = new float[content.size()];
  std::copy(std::begin(content), std::end(content), data);
  auto buffer = new engine::cpu::Buffer(sizeof(float) * content.size(), data);
  pimpl_->shareBuffer(buffer);
}

Tensor::Tensor(std::initializer_list<std::initializer_list<float>> content)
  : pimpl_{nullptr}
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
    pimpl_->shareBuffer(buffer);
  }


}

Tensor::Tensor(std::initializer_list<std::initializer_list<std::initializer_list<float>>> content) {

}

Tensor::~Tensor() {
  bind(nullptr);
}

Tensor& Tensor::operator=(const Tensor& tensor) {
  tensor.assertNotQuery();
  assertNotQuery();

//  print("operator=(const Tensor&)");
  Tensor id = fn::identity(tensor);

  pimpl_->unref();
  pimpl_ = id.pimpl_;
  id.pimpl_ = nullptr;

  return *this;
}

bool Tensor::frame() const {
  assertNotQuery();
  return !pimpl_->frame()->null();
}

const Dtype& Tensor::dtype() const {
  assertNotQuery();
  return pimpl_->dtype();
}

const Shape& Tensor::shape() const {
  assertNotQuery();
  return pimpl_->shape();
}

size_t Tensor::size() const {
  return pimpl_->size();
}

size_t Tensor::rank() const {
  return pimpl_->rank();
}

Tensor Tensor::transpose() const {
  return fn::transpose(*this);
}

Tensor Tensor::t() const {
  return transpose();
}

Tensor Tensor::reshape(const Shape::Reshape& shape) const {
  return fn::reshape(*this, shape);
}

Tensor Tensor::map(const UnaryFn& fn) const {
  return fn(*this);
}

Tensor Tensor::map(const Tensor& linear) const {
  return linear.dot(*this);
}

Tensor Tensor::map(const Tensor& linear, const Tensor& affine) const {
  return linear.dot(*this) + affine;
}

Tensor Tensor::dot(const Tensor& tensor) const {
  return fn::dot(*this, tensor);
}

Tensor Tensor::pow(const Tensor& exponent) const {
  return fn::pow(*this, exponent);
}

Tensor Tensor::nrt(const Tensor& exponent) const {
  return fn::nrt(*this, exponent);
}

Tensor Tensor::norm() const {
  return 1;
}

Tensor Tensor::normalize() const {
  return fn::divide(*this, norm());
}

void* Tensor::data() {
  assertNotQuery();
  if (pimpl_->flow_) {
    throw std::runtime_error("access to tensor buffers inside a flow is forbidden; you have to use a built-in matcha function");
  }

  pimpl_->readData();
  return pimpl_->data();
}

Tensor Tensor::fromOut(engine::Tensor* out) {
  auto tensor = Tensor(out);

  if (!out->flow() && out->source()) {
    out->compute();
  }

  return tensor;
}

Tensor::Tensor(engine::Tensor* pimpl)
  : pimpl_{nullptr}
{
  bind(pimpl);
}

void Tensor::bind(engine::Tensor* tensor) {
  if (pimpl_) pimpl_->unref();
  pimpl_ = tensor;
  if (pimpl_) pimpl_->ref();
}

engine::Flow* Tensor::flowQuery(const UnaryFn& fn) {
  Tensor queryTensor(nullptr);
  try {
    Tensor responseTensor = fn(queryTensor);
  } catch (engine::FlowQueryResponse& r) {
    return r.flow;
  }
  return nullptr;
}

bool Tensor::getFlowQuery() const {
  return pimpl_ == nullptr;
}

void Tensor::assertNotQuery() const {
  if (getFlowQuery()) throw engine::FlowQueryResponse();
}

Tensor zeros(const Shape& shape) {
  auto t = new engine::Tensor(Float, shape);
  auto b = t->writeBuffer();

  auto vals = (float*) b->payload();
  std::fill(vals, vals + shape.size(), 0);

  return Tensor::fromOut(t);
}

Tensor ones(const Shape& shape) {
  auto t = new engine::Tensor(Float, shape);
  auto b = t->writeBuffer();

  auto vals = (float*) b->payload();
  std::fill(vals, vals + shape.size(), 1);

  return Tensor::fromOut(t);
}

Tensor eye(const Shape& shape) {
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

  return Tensor::fromOut(t);
}

}

std::ostream& operator<<(std::ostream& os, const matcha::Tensor& tensor) {
  tensor.assertNotQuery();
  if (!tensor.frame()) {
    os << "Tensor {}" << std::endl;
    return os;
  }

  auto t = tensor.pimpl_;
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
      block = true;
      break;
    case 2:
      block = tensor.shape()[0] > 1;
      break;
    default:
      block = true;
  }

  size_t size = iter.rows * iter.cols;

  os << "Tensor {";
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

