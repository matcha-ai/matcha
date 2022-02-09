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

    Input generateNext() const override;
    void populateNext(Input* input) const override;

    Tensor* openOut() override;
    bool openOut(Tensor* tensor) override;
    bool closeOut(Tensor* tensor) override;
    bool polymorphicOuts() const override;

    void reset() const override;
    void shuffle() const override;
    bool eof() const override;

    size_t size() const override;

    static const NodeLoader* loader();
    const NodeLoader* getLoader() const override;
    void save(std::ostream& os) const override;

    void eval(Tensor* target) override;

  private:
    float m_, sd_;
    uint64_t seed_;

    mutable std::mt19937 source_;
    mutable std::normal_distribution<float> distribution_;
};

}
}
}
