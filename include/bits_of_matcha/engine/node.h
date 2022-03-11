#pragma once

#include "bits_of_matcha/computation.h"
#include "bits_of_matcha/device.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/tensor.h"


#include <vector>

namespace matcha {
class Tensor;
class Device;
}

namespace matcha::engine {

class Tensor;
class PipeBuffers;

class Node {
  public:
    Node();
    Node(std::initializer_list<Tensor*> ins);
    virtual ~Node();

    Tensor* in(int idx);
    Tensor* out(int idx);

    int ins() const;
    int outs() const;

    int inIdx(Tensor* in) const;
    int outIdx(Tensor* out) const;

    virtual void init();
    virtual void run();

    virtual void use(const Device& device);
    virtual const Device::Concrete* device() const;

  protected:
    void createOut(const Frame& frame);
    void createOut(const Dtype& dtype, const Shape& shape);

  protected:
    std::vector<Tensor*> ins_;
    std::vector<Tensor*> outs_;
    std::vector<Buffer*> x_;
    std::vector<Buffer*> y_;
};


}