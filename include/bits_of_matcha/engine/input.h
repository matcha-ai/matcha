#pragma once

#include "bits_of_matcha/engine/object.h"
#include "bits_of_matcha/engine/node.h"


namespace matcha {
namespace device {
  class Buffer;
}

namespace engine {

class Stream;

class Input : public Object {
  public:
    Input(const Dtype& dtype, const Shape& shape);
    Input(const Dtype& dtype, const Shape& shape, const std::vector<std::byte>& content);
    ~Input();

    const Dtype& dtype() const;
    const Shape& shape() const;

    Out* out();

    size_t rank() const;
    size_t size() const;

    template <class T>
    T& at(size_t position);

    void update(Tensor* source);

    void updateStatusChanged(In* in = nullptr) override;
    void prune(Out* out = nullptr) override;

  private:
    device::Buffer* buffer_;
    Out* out_;

};


}
}
