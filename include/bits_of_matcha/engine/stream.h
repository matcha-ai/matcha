#pragma once

#include "bits_of_matcha/engine/object.h"
#include "bits_of_matcha/engine/node.h"


namespace matcha {
class Stream;

namespace engine {

class Input;


class Stream : public Node {
  public:
    Stream();

    virtual bool next() = 0;
    virtual bool seek(size_t pos) = 0;
    virtual size_t tell() const = 0;
    virtual size_t size() const = 0;

    virtual bool eof() const = 0;

    virtual Tensor* open(int idx) = 0;
    virtual void open(int idx, Tensor* tensor) = 0;
    virtual void close(Out* out) = 0;

  public:
    void prune(Out* out = nullptr) override;

  protected:
    void beginOut(Tensor* out);

  protected:
    static Stream* deref(const matcha::Stream* stream);
    static Stream* deref(const matcha::Stream& stream);

};



}
}
