#pragma once

#include "bits_of_matcha/stream.h"
#include "bits_of_matcha/engine/stream.h"

#include <cinttypes>
#include <random>


namespace matcha {
namespace rng {
  Stream normal();
  Stream normal(uint64_t seed);
  Stream normal(float m, float sd);
  Stream normal(float m, float sd, uint64_t seed);
}
}

namespace matcha {
namespace engine {
namespace rng {


class Normal : public Stream {
  public:
    Normal(float m, float sd, uint64_t seed);

    Tensor* open(int idx) override;
    void open(int idx, Tensor* tensor) override;
    void close(Out* out) override;

    void eval(Out* out) override;

    bool next() override;
    bool seek(size_t pos) override;
    size_t tell() const override;
    size_t size() const override;

    bool eof() const override;

  private:
    float m_, sd_;
    uint64_t seed_;

    mutable std::mt19937 source_;
    mutable std::normal_distribution<float> distribution_;
};

}
}
}
