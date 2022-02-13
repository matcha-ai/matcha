#pragma once

#include "bits_of_matcha/engine/relay.h"

#include <cstddef>


namespace matcha {
namespace fn {
  Stream batch(Stream& stream, size_t sizeLimit);
}
}


namespace matcha {
namespace engine {
namespace fn {


class Batch : public Relay {
  public:
    Batch(Stream* source, size_t sizeLimit);
    Batch(matcha::Stream& source, size_t sizeLimit);

    Tensor* open(int idx) override;
    void open(int idx, Tensor* tensor) override;
    void close(Out* out) override;

    bool next() override;
    bool seek(size_t pos) override;
    size_t tell() const override;
    size_t size() const override;

    bool eof() const override;

  private:
    void relayData(Tensor* relay, Tensor* out);

    matcha::Stream ref_;
    Stream* source_;
    size_t sizeLimit_;
    bool limitReached_;
    std::vector<Tensor*> relays_;
    std::vector<unsigned> position_;
    size_t pos_;
    size_t begin_;
};


}
}
}
