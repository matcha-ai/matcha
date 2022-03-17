#pragma once

#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/macros/varargShape.h"
#include "bits_of_matcha/fn.h"


namespace matcha {
  class tensor;
  class Slice;
  class Device;
}

namespace matcha::engine {
  class Tensor;
  class Flow;
  class Node;
  Tensor* deref(const matcha::tensor* tensor);
}

std::ostream& operator<<(std::ostream& os, const matcha::tensor& tensor);


namespace matcha {

class tensor {
public:
  tensor();
  tensor(const Dtype& dtype, const Shape& shape);
  tensor(float content);
  tensor(std::initializer_list<float> content);
  tensor(std::initializer_list<std::initializer_list<float>> content);
  tensor(std::initializer_list<std::initializer_list<std::initializer_list<float>>> content);

  ~tensor();

  bool frame() const;
  const Dtype& dtype() const;
  const Shape& shape() const;
  size_t size() const;
  size_t rank() const;

  Slice operator[](const Shape::Range& range);

  tensor transpose() const;
  tensor t() const;

  tensor reshape(const Shape::Reshape& shape) const;

  template <class... Dims>
  inline tensor reshape(Dims... dims) const {
    return reshape(Shape::Reshape(VARARG_RESHAPE(dims...)));
  }

  tensor map(const UnaryFn& fn) const;
  tensor map(const tensor& linear) const;
  tensor map(const tensor& linear, const tensor& affine) const;
  tensor dot(const tensor& tensor) const;
  tensor pow(const tensor& exponent) const;
  tensor nrt(const tensor& exponent) const;

  tensor norm() const;
  tensor normalize() const;

  tensor(const tensor& tensor);
  tensor& operator=(const tensor& tensor);

  void* data();

  bool getFlowQuery() const;

  static tensor floats(const Shape& shape);
  static tensor full(const Shape& shape, float value);
  static tensor zeros(const Shape& shape);
  static tensor ones(const Shape& shape);
  static tensor eye(const Shape& shape);

  template <class... Dims>
  static inline tensor floats(Dims... dims) {
    return floats(VARARG_SHAPE(dims...));
  }

  template <class... Dims>
  static inline tensor ones(Dims... dims) {
    return ones(VARARG_SHAPE(dims...));
  }

  template <class... Dims>
  static inline tensor zeros(Dims... dims) {
    return zeros(VARARG_SHAPE(dims...));
  }

  template <class... Dims>
  static inline tensor eye(Dims... dims) {
    return eye(VARARG_SHAPE(dims...));
  }

  explicit tensor(engine::Tensor* internal);



private:
  static engine::Flow* flowQuery(const UnaryFn& fn);

  void bind(engine::Tensor* tensor);

  engine::Tensor* internal_;
  void assertNotQuery() const;

  friend engine::Tensor* engine::deref(const matcha::tensor* tensor);
  friend std::ostream& ::operator<<(std::ostream& os, const tensor& tensor);

};

enum {
  Float = Dtype::Float
};

using Tuple = std::vector<tensor>;

tensor floats(const Shape& shape);
tensor full(const Shape& shape, float value);
tensor zeros(const Shape& shape);
tensor ones(const Shape& shape);
tensor eye(const Shape& shape);

template <class... Dims>
inline tensor floats(Dims... dims) {
  return floats(VARARG_SHAPE(dims...));
}

template <class... Dims>
inline tensor ones(Dims... dims) {
  return ones(VARARG_SHAPE(dims...));
}

template <class... Dims>
inline tensor zeros(Dims... dims) {
  return zeros(VARARG_SHAPE(dims...));
}

template <class... Dims>
inline tensor eye(Dims... dims) {
  return eye(VARARG_SHAPE(dims...));
}

}

std::ostream& operator<<(std::ostream& os, const matcha::tensor& tensor);

