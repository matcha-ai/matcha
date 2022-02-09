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
    virtual Input generateNext() const = 0;
    virtual void populateNext(Input* input) const = 0;

    virtual void reset() const = 0;
    virtual void shuffle() const = 0;
    virtual bool eof() const = 0;

    virtual size_t size() const = 0;

  public:
    void require() override;
    void considerPruning() override;

  protected:
    static Stream* deref(const matcha::Stream* stream);
    static Stream* deref(const matcha::Stream& stream);
};



}
}
