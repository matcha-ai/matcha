#pragma once

#include "bits_of_matcha/object.h"

#include <ostream>


namespace matcha {
  class Stream;
  class Tensor;
  class Input;
}

matcha::Stream& operator>>(matcha::Stream& stream, matcha::Tensor& Tensor);
matcha::Stream& operator>>(matcha::Stream& stream, matcha::Input& input);
std::ostream& operator<<(std::ostream& os, const matcha::Stream& stream);


namespace matcha {


class Input;
class Tensor;

namespace engine {
  class Stream;
}


class Stream : public Object {
  public:
    Stream() = default;
    Stream(const Tensor& tensor);
    Stream(const Input& input);

    operator bool() const;

    void reset() const;
    void shuffle() const;

    size_t size() const;

  public:
    static Stream fromObject(engine::Stream* object);

  private:
    Stream(engine::Stream* object, char dummy);
    engine::Stream* object() const;

    friend Stream& ::operator>>(matcha::Stream& stream, matcha::Tensor& tensor);
    friend Stream& ::operator>>(matcha::Stream& stream, matcha::Input& input);
    friend std::ostream& ::operator<<(std::ostream& os, const Stream& stream);

    friend class engine::Stream;
    friend class Tensor;
    friend class Params;
    friend class Input;
    friend class Tuple;
    friend class Flow;

};

}
