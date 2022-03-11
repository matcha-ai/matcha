#pragma once

#include "bits_of_matcha/tensor.h"


namespace matcha {

class Random {
  public:
    Tensor operator()(const Shape& shape);

    template <class... Dims>
    inline Tensor operator()(Dims... dims) {
      return operator()(VARARG_SHAPE(dims...));
    }

    explicit Random(std::function<engine::Tensor* (const Shape& shape)> generator);

  private:
    std::function<engine::Tensor* (const Shape& shape)> generator_;

};

}

namespace matcha::random {

extern Random random;

struct Normal {
  operator Random() const;

  float m = 0;
  float sd = 1;
  uint64_t seed = 0;

  template <class... Dims>
  inline Tensor operator()(Dims... dims) {
    return operator()(VARARG_SHAPE(dims...));
  }

  Tensor operator()(const Shape& shape);
  Random zz_internal_ = Random(*this);

};

struct Uniform {
  operator Random() const;

  float a = 0;
  float b = 1;
  uint64_t seed = 0;

  template <class... Dims>
  inline Tensor operator()(Dims... dims) {
    return operator()(VARARG_SHAPE(dims...));
  }

  Tensor operator()(const Shape& shape);
  Random zz_internal_ = Random(*this);

};

struct Bernoulli {
  operator Random() const;

  float p = 0.5;
  uint64_t seed = 0;
};

struct Poisson {
  operator Random() const;

  float m = 1;
  uint64_t seed = 0;
};

struct Categorical {
  operator Random() const;

  uint64_t seed = 0;
};

using Gaussian = Normal;
using Binomial = Bernoulli;
using Multinomial = Categorical;

}