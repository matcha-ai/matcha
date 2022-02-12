#pragma once

#include "bits_of_matcha/stream.h"
#include "bits_of_matcha/engine/stream.h"

#include <cinttypes>
#include <random>


namespace matcha {
namespace rng {
  Stream uniform();
  Stream uniform(uint64_t seed);
  Stream uniform(float lo, float hi);
  Stream uniform(float lo, float hi, uint64_t seed);
}
}

namespace matcha {
namespace engine {
namespace rng {


class Uniform : public Stream {
  public:
    Uniform(float lo, float hi, uint64_t seed);

    void reset() override;
    void shuffle() override;

    bool eof() const override;
    size_t size() const override;

    Tensor* open() override;
    void open(Tensor* out) override;
    void close(Out* out) override;

    void eval(Out* out) override;

  private:
    float lo_, hi_;
    uint64_t seed_;

    mutable std::mt19937 source_;
    mutable std::uniform_real_distribution<float> distribution_;
};

}
}
}
