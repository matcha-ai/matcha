#pragma once

#include "bits_of_matcha/object.h"

#include <ostream>
#include <functional>


namespace matcha {
  class Stream;
  class Tensor;
  class Input;
}

matcha::Stream& operator>>(matcha::Stream& stream, matcha::Tensor& Tensor);
std::ostream& operator<<(std::ostream& os, const matcha::Stream& stream);


namespace matcha {


class Input;
class Tensor;

namespace engine {
  class Stream;
}


class Stream : public Object {
  public:
    void reset() const;
    void shuffle() const;

    operator bool() const;
    size_t size() const;

    Stream batch(size_t sizeLimit);
    Stream map(std::function<Tensor (const Tensor&)> fn);
    Tensor fold(const Tensor& init, std::function<Tensor (const Tensor&, const Tensor&)> fn);

  public:
    static Stream fromObject(engine::Stream* object);

  private:
    Stream(engine::Stream* object, char dummy);
    engine::Stream* object() const;

    friend Stream& ::operator>>(matcha::Stream& stream, matcha::Tensor& tensor);
    friend std::ostream& ::operator<<(std::ostream& os, const Stream& stream);

    friend class engine::Stream;
    friend class Tensor;
    friend class Params;
    friend class Input;
    friend class Tuple;
    friend class Flow;

};

}
