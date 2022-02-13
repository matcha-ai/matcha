#pragma once

#include "bits_of_matcha/engine/fn.h"
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
    void shuffle() const;

    Tensor operator()();
    Tensor operator()(int idx);

    // actions

    void reset() const;

    bool next();
    bool seek(size_t pos);
    size_t tell() const;
    size_t size() const;

    bool eof() const;
    operator bool() const;

    // operations

    Stream batch(size_t sizeLimit);
    Stream map(UnaryFn fn);
    Tensor fold(const Tensor& init, BinaryFn fn);

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
