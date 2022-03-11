#pragma once

#include "bits_of_matcha/frame.h"
#include "bits_of_matcha/engine/buffer.h"
#include "bits_of_matcha/device.h"

#include <variant>

namespace matcha {
class Tensor;
}

namespace matcha::engine {

struct Node;

class Tensor {
  public:
    explicit Tensor(Frame frame);
    Tensor(const Dtype& dtype, const Shape& shape);
    Tensor();
    ~Tensor();

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
    bool flow() const;

  private:
    Frame frame_;
    Node* source_;
    Buffer* buffer_;
    Buffer* cpuBuffer_;

  private:
    unsigned refs_;
    unsigned reqs_;
    bool flow_;

  private:
    friend class matcha::Tensor;
};


Tensor* deref(const matcha::Tensor& tensor);
Tensor* deref(const matcha::Tensor* tensor);

}