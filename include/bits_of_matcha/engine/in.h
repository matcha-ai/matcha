#pragma once

#include <cstddef>


namespace matcha {

class Dtype;
class Shape;

namespace device {
  class Buffer;
}

namespace engine {

class Out;
class Object;
class Status;

class In {
  public:
    // from target

    In(Out* source, Object* target, unsigned id);
    ~In();

    unsigned id() const;

    const Dtype& dtype() const;
    const Shape& shape() const;
    const Status& status() const;

    size_t rank() const;
    size_t size() const;

    void eval();

    device::Buffer* buffer();

  public:
    // from source

    void dataStatusChanged();
    void updateStatusChanged();
    void bufferChanged();

  private:
    Out* source_;
    Object* target_;
    unsigned id_;

};


}
}
