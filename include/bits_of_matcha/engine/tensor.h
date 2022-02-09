#pragma once

#include "bits_of_matcha/engine/object.h"
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
    Tensor(Node* in, device::Buffer* buffer);
    Tensor(device::Buffer* buffer);

    const Dtype& dtype() const;
    const Shape& shape() const;

    size_t rank() const;
    size_t size() const;

    void eval();
    void require();
    bool required() const;

    void bindOut(Node* out);
    void unbindOut(Node* out);
    void bindIn(Node* in, unsigned evalId);
    void unbindIn(Node* in);

    unsigned edgeId() const;
    void setEdgeId(unsigned edgeId) const;

    bool ready() const;
    void setReady(bool ready) const;

    device::Buffer* buffer();
    const device::Buffer* buffer() const;

    bool hasData() const;
    const std::byte* getData() const;

    void setBuffer(device::Buffer* buffer);

  public:
    void considerPruning() override;

  private:
    Dtype dtype_;
    Shape shape_;

    device::Buffer* buffer_;
    device::Buffer* cpuBuffer_;
    mutable bool ready_;

    mutable Node* in_;
    mutable unsigned edgeId_;
    mutable std::set<Node*> outs_;

    mutable bool required_;
    void unrequire() const;

    friend class Flow;
    friend class FlowSaver;

};


}
}
