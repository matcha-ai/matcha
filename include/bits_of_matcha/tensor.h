#pragma once

#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/macros/vararg_shape.h"

#include <iostream>

#define MATCHA_TENSOR_API alignas(void*)


namespace matcha {

/**
 * multidimensional array
 * matcha primitive
 */
class MATCHA_TENSOR_API tensor {
public:

  /**
   * @return the tensor frame
   */
   const Frame& frame() const;

  /**
   * @return the tensor dtype
   */
  const Dtype& dtype() const;

  /**
   * @return the tensor shape
   */
  const Shape& shape() const;

  /**
   * @returns tensor transpose
   */
  tensor transpose() const;

  /**
   * @returns tensor transpose
   */
  tensor t() const;

  /**
   * @return the dot product of two tensors
   */
  tensor dot(const tensor& b);

  /**
   * @return the concatenation of two tensors
   */
  tensor cat(const tensor& b);

  tensor pow(const tensor& b);

public:
  tensor();
  tensor(float scalar);
  ~tensor() = default;

  /**
   * @return tensor of given shape filled with specified value
   */
  static tensor full(float value, const Shape& shape);

  /**
   * The zero tensor
   * @return tensor of specified shape full of zeros
   */
  static tensor zeros(const Shape& shape);

  /**
   * The ones tensor
   * @return tensor of specified shape full of ones
   */
  static tensor ones(const Shape& shape);

  /**
   * The identity tensor
   * @return slice of the identity matrix of given shape
   */
  static tensor eye(const Shape& shape);

  /**
   * The zero tensor
   * @return tensor of specified shape full of zeros
   */
  template <class... Dims>
  static inline tensor zeros(Dims... dims) { return zeros(VARARG_SHAPE(dims...)); }

  /**
   * The ones tensor
   * @return tensor of specified shape full of ones
   */
  template <class... Dims>
  static inline tensor ones(Dims... dims) { return ones(VARARG_SHAPE(dims...)); }

  /**
   * The identity tensor
   * @return slice of the identity matrix of given shape
   */
  template <class... Dims>
  static inline tensor eye(Dims... dims) { return eye(VARARG_SHAPE(dims...)); }

public:
  tensor& operator=(const tensor& other);

public:
  /**
   * tensor data
   * @return pointer to tensor data
   */
  void* data();

private:
  void* internal_;
};

}

std::ostream& operator<<(std::ostream& os, const matcha::tensor& t);