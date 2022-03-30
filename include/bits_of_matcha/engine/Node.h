#pragma once

#include "bits_of_matcha/Computation.h"
#include "bits_of_matcha/Device.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/Tensor.h"

#include <vector>


namespace matcha {
class tensor;
class Device;
}

namespace matcha::engine {

class Tensor;
class NodeBackward;

class Node {
public:
  Node();
  Node(std::initializer_list<Tensor*> ins);
  virtual ~Node();

  Tensor* in(int idx);
  Tensor* out(int idx);

  int degIn() const;
  int degOut() const;

  const std::vector<Tensor*>& ins() const;
  const std::vector<Tensor*>& outs() const;

  int inIdx(Tensor* in) const;
  int outIdx(Tensor* out) const;

  virtual void init();
  virtual void run();

  virtual void use(const Device& device);
  virtual const Device::Concrete* device() const;

  virtual bool sideEffect() const;

  bool flow() const;
  int ctxId() const;
  void setCtxId(int ctxId);

  using ArgPartials = std::vector<int>;
  NodeBackward* createBackward(const std::vector<Tensor*>& chainIns, const ArgPartials& argPartials);

protected:
  void createOut(const Frame& frame);
  void createOut(const Dtype& dtype, const Shape& shape);

protected:
  std::vector<Tensor*> ins_;
  std::vector<Tensor*> outs_;
  std::vector<Buffer*> x_;
  std::vector<Buffer*> y_;


private:
  int ctxId_;
  bool flow_;
};


}