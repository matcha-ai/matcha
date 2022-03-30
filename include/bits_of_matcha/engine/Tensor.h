#pragma once

#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/engine/Buffer.h"
#include "bits_of_matcha/Device.h"

#include <variant>


namespace matcha {
class tensor;
}

namespace matcha::engine {

struct Node;

class Tensor {
public:
  explicit Tensor(Frame frame);
  Tensor(const Dtype& dtype, const Shape& shape);
  Tensor();
  ~Tensor();

  static Tensor* full(float value, const Shape& shape);

  const Frame* frame() const;
  const Dtype& dtype() const;
  const Shape& shape() const;
  size_t size() const;
  size_t rank() const;
  size_t bytes() const;

  void readData();
  void* data();

  Buffer* buffer();
  void shareBuffer(Buffer* buffer);
  void shareBuffer(Tensor* tensor);
  void stealBuffer(Tensor* tensor);
  Buffer* writeBuffer(const Device::Concrete& device = CPU);

  Node* source();
  void setSource(Node* source);
  void compute();

  const Device::Concrete* device() const;
  bool uses(const Device::Concrete& device) const;
  bool uses(const Device::Concrete* device) const;

  void ref();
  void unref();
  void req();
  void unreq();

  unsigned refs() const;
  unsigned reqs() const;

  bool eager() const;
  bool lazy() const;

  enum {
    Untraced = 0,
    Constant = 1,
    Variable = 2,
    Gradable = 3
  };

  unsigned mode() const;
  void setMode(unsigned mode);

  void assign(Tensor* source);
  void update(Tensor* source);

  int ctxId() const;
  void setCtxId(int ctxId);

private:
  int ctxId_;

  Frame frame_;
  Node* source_;

  Buffer* bufferInternal_;
  Buffer* bufferExternal_;

  unsigned refs_;
  unsigned reqs_;
  bool flow_;

  unsigned mode_;

private:
  friend class matcha::tensor;
};


Tensor* deref(const matcha::tensor& tensor);
Tensor* deref(const matcha::tensor* tensor);

}