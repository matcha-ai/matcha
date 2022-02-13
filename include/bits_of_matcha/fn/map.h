#pragma once

#include "bits_of_matcha/engine/fn.h"
#include "bits_of_matcha/engine/relay.h"

#include <functional>


namespace matcha {

class Stream;

namespace fn {

Stream map(Stream& stream, UnaryFn fn);

}

namespace engine {

class Stream;

namespace fn {

class Map : public Relay {
  public:
    Map(Stream* stream, UnaryFn fn);
    Map(matcha::Stream& stream, UnaryFn fn);

    Tensor* open(int idx) override;
    void open(int idx, Tensor* tensor) override;
    void close(Out* out) override;

    bool next() override;
    bool seek(size_t pos) override;
    size_t tell() const override;
    size_t size() const override;

    bool eof() const override;

  private:
    matcha::Stream ref_;
    Stream* source_;
    UnaryFn fn_;

    std::vector<Tensor*> relays_;
    std::vector<Tensor*> results_;

    void relayData(Tensor* relay, Tensor* result, Tensor* target);

};

}
}
}
