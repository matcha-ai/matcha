#pragma once

#include "bits_of_matcha/frame.h"
#include "bits_of_matcha/varargShape.h"
#include "bits_of_matcha/fn.h"


namespace matcha {

class Tensor;
class Params;
class Slice;

class Device;

namespace engine {
class Tensor;

class FlowContext;

class Flow;

class Node;

Tensor* deref(const matcha::Tensor* tensor);
}

}

std::ostream& operator<<(std::ostream& os, const matcha::Tensor& tensor);

namespace matcha {

class Tensor {
public:
  Tensor();
  Tensor(const Dtype& dtype, const Shape& shape);
  Tensor(float content);
  Tensor(std::initializer_list<float> content);
  Tensor(std::initializer_list<std::initializer_list<float>> content);
  Tensor(std::initializer_list<std::initializer_list<std::initializer_list<float>>> content);

//    Tensor(const Params& params);

  ~Tensor();

  bool frame() const;
  const Dtype& dtype() const;
  const Shape& shape() const;
  size_t size() const;
  size_t rank() const;

  Slice operator[](int idx);

  Tensor transpose() const;
  Tensor t() const;

  Tensor reshape(const Shape::Reshape& shape) const;

  template <class... Dims>
  inline Tensor reshape(Dims... dims) const {
    return reshape(Shape::Reshape(VARARG_RESHAPE(dims...)));
  }

  Tensor map(const UnaryFn& fn) const;
  Tensor map(const Tensor& linear) const;
  Tensor map(const Tensor& linear, const Tensor& affine) const;
  Tensor dot(const Tensor& tensor) const;
  Tensor pow(const Tensor& exponent) const;
  Tensor nrt(const Tensor& exponent) const;

  Tensor norm() const;
  Tensor normalize() const;

  Tensor(const Tensor& tensor);
  Tensor& operator=(const Tensor& tensor);

  void* data();
  float* floats();

  bool getFlowQuery() const;

  static Tensor fromOut(engine::Tensor* out);

private:
  static engine::Flow* flowQuery(const UnaryFn& fn);

  explicit Tensor(engine::Tensor* pimpl);
  void bind(engine::Tensor* tensor);

  engine::Tensor* pimpl_;
  void assertNotQuery() const;

  friend class engine::FlowContext;
  friend class engine::Node;
  friend class Flow;
  friend engine::Tensor* engine::deref(const matcha::Tensor* tensor);
  friend std::ostream& ::operator<<(std::ostream& os, const Tensor& tensor);

};

enum {
  Float = Dtype::Float
};

Tensor floats(const Shape& shape);
Tensor zeros(const Shape& shape);
Tensor ones(const Shape& shape);
Tensor eye(const Shape& shape);

template <class... Dims>
inline Tensor floats(Dims... dims) {
  return floats(VARARG_SHAPE(dims...));
}

template <class... Dims>
inline Tensor ones(Dims... dims) {
  return ones(VARARG_SHAPE(dims...));
}

template <class... Dims>
inline Tensor zeros(Dims... dims) {
  return zeros(VARARG_SHAPE(dims...));
}

template <class... Dims>
inline Tensor eye(Dims... dims) {
  return eye(VARARG_SHAPE(dims...));
}

}

std::ostream& operator<<(std::ostream& os, const matcha::Tensor& tensor);

