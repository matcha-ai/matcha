#pragma once

#include "bits_of_matcha/dtype.h"
#include "bits_of_matcha/shape.h"

#include <vector>
#include <set>


namespace matcha {

class Dtype;
class Shape;

namespace device {
  class Buffer;
}

namespace engine {

class Object;
class Status;
class In;

class Out {
  public:
    // from target

    void bind(In* target);
    void unbind(In* target);

    void prepare();
    void eval();

    const Dtype& dtype() const;
    const Shape& shape() const;

    size_t rank() const;
    size_t size() const;

    const Status& status() const;

    device::Buffer* buffer();

  public:
    // from source

    Out(const Dtype& dtype, const Shape& shape, Object* source, unsigned id);

    unsigned id() const;
    void setId(unsigned id);
    bool linked() const;

    void dataStatusChanged();
    void updateStatusChanged();
    void setBuffer(device::Buffer* buffer);

  private:
    Object* source_;
    std::set<In*> targets_;
    unsigned id_;

    Dtype dtype_;
    Shape shape_;

    device::Buffer* buffer_;
};

}
}
