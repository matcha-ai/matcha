#pragma once

#include "bits_of_matcha/engine/object.h"
#include "bits_of_matcha/engine/node.h"


namespace matcha {
namespace device {
  class Buffer;
}

namespace engine {

class Stream;

class Input : public Node {
  public:
    Input(const Dtype& dtype, const Shape& shape);
    Input(const Dtype& dtype, const Shape& shape, const std::vector<std::byte>& content);

    Input(const Stream& stream);

    const Dtype& dtype() const;
    const Shape& shape() const;

    size_t rank() const;
    size_t size() const;

    void eval(Tensor* target) override;
    void require() override;

    template <class T>
    T& at(size_t position);

    static const NodeLoader* loader();
    const NodeLoader* getLoader() const override;
    void save(std::ostream& os) const override;

  private:
    device::Buffer* buffer_;

};


}
}
