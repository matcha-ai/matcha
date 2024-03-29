#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/engine/tensor/RefReqCounted.h"
#include "bits_of_matcha/engine/memory/Block.h"
#include "Buffer.h"

#include <iostream>
#include <complex>


namespace matcha::engine {

class Block;
class Op;

class Tensor : public RefReqCounted {
public:
  explicit Tensor(const Dtype& dtype, const Shape& shape);
  explicit Tensor(const Frame& frame);
  virtual ~Tensor();

  const Frame& frame() const;
  const Dtype& dtype() const;
  const Shape& shape() const;
  size_t rank() const;
  size_t size() const;
  size_t bytes() const;

  Buffer& buffer();
  Buffer& malloc();
  Buffer& free();
  Buffer& share(Buffer& buffer);
  Buffer& share(Tensor* tensor);

  void* readData();

  Op* op() const;
  void setOp(Op* op);

private:
  Frame frame_;
  Buffer buffer_;
  Op* op_;
};

tensor ref(Tensor* internal);
std::vector<tensor> ref(const std::vector<Tensor*>& internals);

Tensor* deref(const tensor& external);
Tensor* deref(const tensor* external);

std::vector<Tensor*> deref(const std::vector<tensor>& externals);
std::vector<Tensor*> deref(const std::vector<tensor*>& externals);

Tensor* unref(tensor& external);
Tensor* unref(tensor* external);

std::vector<Tensor*> unref(std::vector<tensor>& externals);
std::vector<Tensor*> unref(std::vector<tensor*>& externals);

}