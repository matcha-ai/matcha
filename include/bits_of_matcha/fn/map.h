#pragma once

#include "bits_of_matcha/engine/stream.h"
#include "bits_of_matcha/stream.h"

#include <functional>


namespace matcha {

class Stream;

namespace fn {

Stream map(Stream& stream, std::function<Tensor (const Tensor&)> fn);

}

namespace engine {

class Stream;

namespace fn {

class Map : public Stream {
  public:
    Map(Stream* stream, std::function<matcha::Tensor (const matcha::Tensor&)> fn);
    Map(matcha::Stream& stream, std::function<matcha::Tensor (const matcha::Tensor&)> fn);

    void reset() override;
    void shuffle() override;

    bool eof() const override;
    size_t size() const override;

    Tensor* open() override;
    void open(Tensor* out) override;
    void close(Out* out) override;

    void eval(Out* out) override;

  private:
    matcha::Stream ref_;
    Stream* source_;
    std::function<matcha::Tensor (const matcha::Tensor&)> fn_;

    std::vector<Tensor*> relays_;
    std::vector<Tensor*> results_;

    void relayData(Tensor* relay, Tensor* result, Tensor* target);

};

}
}
}
