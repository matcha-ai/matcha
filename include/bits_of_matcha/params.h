#pragma once

#include "bits_of_matcha/object.h"

#include <cstddef>
#include <ostream>


namespace matcha {
  class Params;
}


std::ostream& operator<<(std::ostream& os, const matcha::Params& tensor);


namespace matcha {

class Dtype;
class Shape;

class Tensor;
class Input;
class Stream;

namespace engine {
  class Params;
}


class Params : public Object {
  public:
    Params(const Dtype& dtype, const Shape& shape);
    Params(const Dtype& dtype, const Shape& shape, const Tensor& init);
    Params(const Dtype& dtype, const Shape& shape, const Stream& init);
    Params(const Tensor& tensor);
    Params(const Input& init);
    Params(const Stream& init);

    const Params& operator=(const Tensor& tensor);

    const Dtype& dtype() const;
    const Shape& shape() const;

    size_t rank() const;
    size_t size() const;

    void update(const Tensor& value);

  public:
    static Params fromObject(engine::Params* object);

  private:
    engine::Params* object() const;
    Params(engine::Params* object, char dummy);

    friend std::ostream& ::operator<<(std::ostream& os, const Params& params);
    friend class Tensor;

};


}
