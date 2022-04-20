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

tensor ref(Tensor* internal);
std::vector<tensor> ref(const std::vector<Tensor*> internals);

Tensor* deref(const tensor& external);
Tensor* deref(const tensor* external);

std::vector<Tensor*> deref(const std::vector<tensor>& externals);
std::vector<Tensor*> deref(const std::vector<tensor*>& externals);

Tensor* unref(tensor& external);
Tensor* unref(tensor* external);

}