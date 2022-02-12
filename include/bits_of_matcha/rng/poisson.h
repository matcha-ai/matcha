#pragma once

#include "bits_of_matcha/stream.h"
#include "bits_of_matcha/engine/stream.h"

#include <cinttypes>
#include <random>


namespace matcha {
namespace rng {
  Stream poisson();
  Stream poisson(float m);
  Stream poisson(float m, uint64_t seed);
}

namespace engine {
namespace rng {

class Poisson : public Stream {
  public:
    Poisson(float m, uint64_t seed);

    void reset() override;
    void shuffle() override;

    bool eof() const override;
    size_t size() const override;

    Tensor* open() override;
    void open(Tensor* out) override;
    void close(Out* out) override;

    void eval(Out* out) override;

  private:
    float m_, sd_;
    uint64_t seed_;

    mutable std::mt19937 source_;
    mutable std::poisson_distribution<int> distribution_;
};

}
}
}
