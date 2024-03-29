#pragma once

#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/View.h"
#include "bits_of_matcha/macros/vararg_shape.h"
#include "bits_of_matcha/savers/SaveSpec.h"

#include <iostream>
#include <complex>

#define MATCHA_TENSOR_API alignas(void*)


namespace matcha {

class View;

/**
 * multidimensional array
 * matcha primitive
 */
class MATCHA_TENSOR_API tensor final {
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

  size_t size() const;
  size_t rank() const;

  /**
   * @returns tensor reshaped to specified dimensions
   * @see matcha::reshape
   */
  template <class... Dims>
  tensor reshape(Dims... dims) const { return reshape(VARARG_RESHAPE(dims...)); };

  /**
   * @returns tensor reshaped to specified dimensions
   * @see matcha::reshape
   */
  tensor reshape(const Shape::Reshape& dims) const;

  /**
   * @returns tensor transpose
   */
  tensor transpose() const;

  /**
   * @returns tensor transpose
   */
  tensor t() const;

  /**
   * @param b the second tensor
   * @return the matrix multiplication of the two tensors
   */
  tensor matmul(const tensor& b) const;

  /**
   * @param b the second tensor
   * @return concatenation of two tensors
   */
  tensor cat(const tensor& b) const;

  /**
   * @param b exponent
   * @return tensor to the power of b
   */
  tensor power(const tensor& b) const;

  /**
   * @param dtype target dtype
   * @return tensor of specified dtype
   */
  tensor cast(const Dtype& dtype) const;

  auto operator[](const tensor& idxs) -> View;
  auto operator[](const tensor& idxs) const -> const View;

  View* begin();
  View* end();

  explicit operator bool() const;

public:
  tensor() noexcept;

  tensor(float scalar);
  tensor(double scalar);

  tensor(int8_t scalar);
  tensor(int16_t scalar);
  tensor(int32_t scalar);
  tensor(int64_t scalar);

  tensor(uint8_t scalar);
  tensor(uint16_t scalar);
  tensor(uint32_t scalar);
  tensor(uint64_t scalar);

  tensor(std::complex<int32_t> scalar);
  tensor(std::complex<uint32_t> scalar);
  tensor(std::complex<float> scalar);
  tensor(std::complex<double> scalar);

  tensor(bool scalar);

  static tensor full(float value, const Shape& shape);
  static tensor zeros(const Shape& shape);
  static tensor ones(const Shape& shape);
  static tensor eye(const Shape& shape);

  static tensor blob(const void* data, const Frame& frame);
  static tensor blob(const void* data, const Dtype& dtype, const Shape& shape);
  static tensor blob(const float* data, const Shape& shape);
  static tensor blob(const std::vector<float>& data, const Shape& shape);
  static tensor blob(const std::vector<float>& data);

  template <class... Dims>
  static inline tensor zeros(Dims... dims) { return zeros(VARARG_SHAPE(dims...)); }

  template <class... Dims>
  static inline tensor ones(Dims... dims) { return ones(VARARG_SHAPE(dims...)); }

  template <class... Dims>
  static inline tensor eye(Dims... dims) { return eye(VARARG_SHAPE(dims...)); }

public:
  tensor(const tensor& other);
  tensor(tensor&& other) noexcept;
  tensor& operator=(const tensor& other);
  tensor& operator=(tensor&& other) noexcept;
  tensor& assign(const tensor& other);
  ~tensor();

  tensor(std::initializer_list<float> vector);
  tensor(std::initializer_list<std::initializer_list<float>> matrix);
  tensor(std::initializer_list<std::initializer_list<std::initializer_list<float>>> rank3tensors);
  tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<float>>>> rank4tensors);

public:
  /**
   * @brief tensor data
   * @return pointer to tensor data
   * @warning forbidden inside the `Flow`
   */
  void* data();

  /**
   * @brief saves tensor in given format
   */
  void save(const std::string& file, SaveSpec spec = {});

private:
  friend class Engine;
  explicit tensor(void* internal);
  void* internal_;
};


template <class... Dims>
static inline tensor zeros(Dims... dims) { return tensor::zeros(dims...); }

template <class... Dims>
static inline tensor ones(Dims... dims) { return tensor::ones(dims...); }

template <class... Dims>
static inline tensor eye(Dims... dims) { return tensor::eye(dims...); }

static tensor full(float value, const Shape& shape) { return tensor::full(value, shape); }
static tensor zeros(const Shape& shape) { return tensor::zeros(shape); }
static tensor ones(const Shape& shape) { return tensor::ones(shape); }
static tensor eye(const Shape& shape) { return tensor::eye(shape); }

static tensor blob(void* data, const Frame& frame) { return tensor::blob(data, frame); }
static tensor blob(void* data, const Dtype& dtype, const Shape& shape) { return tensor::blob(data, dtype, shape); }
static tensor blob(float* data, const Shape& frame) { return tensor::blob(data, frame); };
static tensor blob(const std::vector<float>& data, const Shape& shape) { return tensor::blob(data, shape); };
static tensor blob(const std::vector<float>& data) { return tensor::blob(data); };

std::ostream& operator<<(std::ostream& os, const matcha::tensor& t);
}

//std::ostream& operator<<(std::ostream& os, const matcha::tensor& t);