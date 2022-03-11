#pragma once

#include "bits_of_matcha/engine/elementwiseBinary.h"


namespace matcha::fn {

Tensor eq(const Tensor& a, const Tensor& b);
Tensor ne(const Tensor& a, const Tensor& b);
Tensor lt(const Tensor& a, const Tensor& b);
Tensor gt(const Tensor& a, const Tensor& b);
Tensor le(const Tensor& a, const Tensor& b);
Tensor ge(const Tensor& a, const Tensor& b);

Tensor maxBetween(const Tensor& a, const Tensor& b);
Tensor minBetween(const Tensor& a, const Tensor& b);
Tensor max(const Tensor& a, const Tensor& b);
Tensor min(const Tensor& a, const Tensor& b);

}


matcha::Tensor operator==(const matcha::Tensor& a, const matcha::Tensor& b);
matcha::Tensor operator!=(const matcha::Tensor& a, const matcha::Tensor& b);
matcha::Tensor operator>(const matcha::Tensor& a, const matcha::Tensor& b);
matcha::Tensor operator<(const matcha::Tensor& a, const matcha::Tensor& b);
matcha::Tensor operator>=(const matcha::Tensor& a, const matcha::Tensor& b);
matcha::Tensor operator<=(const matcha::Tensor& a, const matcha::Tensor& b);


namespace matcha::engine::fn {

class Eq : public ElementwiseBinary {
  public:
    Eq(Tensor* a, Tensor* b);
    void run() override;
};

class Ne : public ElementwiseBinary {
  public:
    Ne(Tensor* a, Tensor* b);
    void run() override;
};

class Lt : public ElementwiseBinary {
  public:
    Lt(Tensor* a, Tensor* b);
    void run() override;
};

class Le : public ElementwiseBinary {
  public:
    Le(Tensor* a, Tensor* b);
    void run() override;
};

class MaxBetween : public ElementwiseBinary {
  public:
    MaxBetween(Tensor* a, Tensor* b);
    void run() override;
};

class MinBetween : public ElementwiseBinary {
  public:
    MinBetween(Tensor* a, Tensor* b);
    void run() override;
};

}