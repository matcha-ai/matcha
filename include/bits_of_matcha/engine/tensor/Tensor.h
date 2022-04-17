#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/engine/tensor/RefReqCounted.h"
#include "bits_of_matcha/engine/tensor/TensorCtx.h"
#include "bits_of_matcha/engine/memory/Buffer.h"

#include <iostream>


namespace matcha::engine {

class Buffer;
class Op;

class Tensor : RefReqCounted {
public:
  explicit Tensor(const Dtype& dtype, const Shape& shape, Op* op = nullptr);
  explicit Tensor(const Frame& frame, Op* op = nullptr);
  virtual ~Tensor();

  const Frame& frame() const;
  const Dtype& dtype() const;
  const Shape& shape() const;
  size_t rank() const;
  size_t size() const;
  size_t bytes() const;

  Buffer* malloc();
  Buffer* buffer();
  void shareBuffer(Buffer* buffer);
  void shareBuffer(Tensor* tensor);

  TensorCtx& ctx();
  Op* op();
  void setOp(Op* op);

  void repr(std::ostream& os);

private:
  Frame frame_;
  Buffer* buffer_;
  TensorCtx ctx_;
  Op* op_;
};

Tensor* full(float value, const Shape& shape);
Tensor* zeros(const Shape& shape);
Tensor* ones(const Shape& shape);
Tensor* eye(const Shape& shape);

tensor ref(Tensor* internal);
Tensor* deref(const tensor& internal);
Tensor* deref(const tensor* internal);
Tensor* unref(tensor& internal);
Tensor* unref(tensor* internal);

}