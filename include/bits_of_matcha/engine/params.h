#pragma once

#include "bits_of_matcha/engine/object.h"
#include "bits_of_matcha/engine/node.h"
#include "bits_of_matcha/dtype.h"
#include "bits_of_matcha/shape.h"


namespace matcha {
namespace device {
  class Buffer;
}

namespace engine {

class Tensor;
class Input;
class Stream;


class Params : public Object {
  public:
    Params(const Dtype& dtype, const Shape& shape);
    Params(const Dtype& dtype, const Shape& shape, Tensor* init);
    Params(const Dtype& dtype, const Shape& shape, Stream* init);
    Params(const Dtype& dtype, const Shape& shape, const std::byte* data);
    Params(Tensor* tensor);
    Params(Input* init);
    Params(Stream* init);

    const Dtype& dtype() const;
    const Shape& shape() const;

    size_t rank() const;
    size_t size() const;

    void update(Tensor* source);

    Out* out();

    void prune(Out* out = nullptr) override;

  private:
    Dtype dtype_;
    Shape shape_;

    Out* out_;

  private:
    device::Buffer* buffer_;

};


}
}
