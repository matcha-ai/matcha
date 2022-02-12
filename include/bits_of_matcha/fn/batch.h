#pragma once

#include "bits_of_matcha/stream.h"
#include "bits_of_matcha/engine/stream.h"

#include <cstddef>


namespace matcha {
namespace fn {
  Stream batch(Stream& stream, size_t sizeLimit);
}
}


namespace matcha {
namespace engine {
namespace fn {


class Batch : public Stream {
  public:
    Batch(Stream* source, size_t sizeLimit);
    Batch(matcha::Stream& source, size_t sizeLimit);

    void reset() override;
    void shuffle() override;

    bool eof() const override;
    size_t size() const override;

    Tensor* open() override;
    void open(Tensor* out) override;
    void close(Out* out) override;

    void eval(Out* out) override;

  private:
    void relayData(Tensor* relay, Tensor* out);

    matcha::Stream ref_;
    Stream* source_;
    size_t sizeLimit_;
    bool limitReached_;
    std::vector<Tensor*> relays_;
    std::vector<unsigned> position_;
};


}
}
}
