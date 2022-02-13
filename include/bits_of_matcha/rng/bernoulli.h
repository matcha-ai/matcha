#pragma once

#include "bits_of_matcha/engine/stream.h"
#include "bits_of_matcha/engine/nodeloader.h"

#include <cinttypes>
#include <random>


namespace matcha {
class Stream;

namespace rng {
  Stream bernoulli();
  Stream bernoulli(float p);
  Stream bernoulli(float p, uint64_t seed);
}

namespace engine {
namespace rng {


class Bernoulli : public Stream {
  public:
    Bernoulli(float p, uint64_t seed);

    Tensor* open(int idx) override;
    void open(int idx, Tensor* out) override;
    void close(Out* out) override;

    void eval(Out* out) override;

    bool next() override;
    bool seek(size_t pos) override;
    size_t tell() const override;
    size_t size() const override;

    bool eof() const override;

  private:
    float p_;
    uint64_t seed_;

    mutable std::mt19937 source_;
    mutable std::bernoulli_distribution distribution_;
};

}
}
}
