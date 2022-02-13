#pragma once

#include "bits_of_matcha/engine/object.h"
#include "bits_of_matcha/engine/status.h"
#include "bits_of_matcha/engine/in.h"
#include "bits_of_matcha/engine/out.h"
#include "bits_of_matcha/dtype.h"
#include "bits_of_matcha/shape.h"

#include <iostream>
#include <set>


namespace matcha {
namespace device {
  class Buffer;

  namespace cpu {
    class Buffer;
  }
}

class Input;
class Tensor;

namespace engine {

class Tensor;
class Params;
class Stream;
class Input;
class Node;

class Tensor : public Object {
  public:
    Tensor(const Dtype& dtype, const Shape& shape);
    Tensor(Out* source);

    const Dtype& dtype() const;
    const Shape& shape() const;

    size_t rank() const;
    size_t size() const;

    void dataStatusChanged(In* in) override;
    void updateStatusChanged(In* in = nullptr) override;
    void bufferChanged(In* in) override;
    void eval(Out* out = nullptr) override;
    void prune(Out* out = nullptr) override;

    In* in();
    Out* out();

    void subst(Out* source);
    void subst();

    device::Buffer* buffer();
    const device::Buffer* buffer() const;

    void* data();

    void setBuffer(device::Buffer* buffer);

  private:
    Dtype dtype_;
    Shape shape_;

    device::Buffer* buffer_;
    mutable device::Buffer* cpuBuffer_;

    void unrequire() const;

    friend class Flow;
    friend class FlowSaver;

  private:
    In* in_;
    Out* out_;

};


}
}
