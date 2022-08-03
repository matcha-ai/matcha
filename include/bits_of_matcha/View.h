#pragma once

#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/tensor.h"


namespace matcha {

class tensor;

class View {
public:
  operator tensor() const;

  View& operator=(const tensor& t);

  /**
   * @return the tensor frame
   */
  Frame frame() const;

  /**
   * @return the tensor dtype
   */
  Dtype dtype() const;

  /**
   * @return the tensor shape
   */
  Shape shape() const;

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
   * @return the matrix multiplication of the tensors
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

  View operator[](const Shape::Range& range);
  View operator[](const tensor& idxs);

private:
  tensor* tensor_;
  std::vector<Shape::Range> ranges_;
};

}