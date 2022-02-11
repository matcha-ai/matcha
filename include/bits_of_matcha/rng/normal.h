#pragma once

#include "bits_of_matcha/engine/stream.h"
#include "bits_of_matcha/engine/nodeloader.h"

#include <cinttypes>
#include <random>


namespace matcha {
class Stream;

namespace rng {
  Stream normal();
  Stream normal(uint64_t seed);
  Stream normal(float m, float sd);
  Stream normal(float m, float sd, uint64_t seed);
}

namespace engine {
namespace rng {


class Normal : public Stream {
  public:
    Normal(float m, float sd, uint64_t seed);

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
    mutable std::normal_distribution<float> distribution_;
};

}
}
}
