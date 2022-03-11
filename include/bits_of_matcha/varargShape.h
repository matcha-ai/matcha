#pragma once

#include "bits_of_matcha/shape.h"


namespace matcha {

template <class... Dims>
inline Shape VARARG_SHAPE(Dims...);

template <class... Dims>
inline Shape::Reshape VARARG_RESHAPE(Dims...);



static thread_local std::vector<int> VARARG_SHAPE_BUFFER;

template <class... Dims>
inline void BUILD_VARARG_DIMS(Dims... dims);


template <class Dim, class... Dims>
inline void BUILD_VARARG_DIMS(Dim dim, Dims... dims) {
  VARARG_SHAPE_BUFFER.push_back(dim);
  BUILD_VARARG_DIMS(dims...);
}

template <>
inline void BUILD_VARARG_DIMS() {

}

template <class... Dims>
inline Shape VARARG_SHAPE(Dims... dims) {
  VARARG_SHAPE_BUFFER.clear();
  BUILD_VARARG_DIMS(dims...);
  std::vector<unsigned> shape;
  shape.reserve(VARARG_SHAPE_BUFFER.size());
  for (int dim: VARARG_SHAPE_BUFFER) {
    if (dim <= 0) throw std::invalid_argument("Shape dims must be positive");
    shape.push_back(dim);
  }
  return shape;
}

template <class... Dims>
inline Shape::Reshape VARARG_RESHAPE(Dims... dims) {
  VARARG_SHAPE_BUFFER.clear();
  BUILD_VARARG_DIMS(dims...);
  return Shape::Reshape(VARARG_SHAPE_BUFFER);
}

template <class Dims>
inline Shape::Reshape VARARG_RESHAPE(const Dims& dims) {
  return Shape::Reshape(dims);
}

template <>
inline Shape::Reshape VARARG_RESHAPE(const int& dim) {
  return Shape::Reshape({dim});
}

template <>
inline Shape::Reshape VARARG_RESHAPE(const Shape& shape) {
  return Shape::Reshape(shape);
}

}