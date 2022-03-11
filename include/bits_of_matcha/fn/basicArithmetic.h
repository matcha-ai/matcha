#pragma once

#include "bits_of_matcha/engine/elementwiseBinary.h"
#include "bits_of_matcha/engine/elementwiseUnary.h"


namespace matcha::fn {

Tensor add(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor divide(const Tensor& a, const Tensor& b);
Tensor negative(const Tensor& a);
Tensor abs(const Tensor& a);

}


matcha::Tensor operator+(const matcha::Tensor& a, const matcha::Tensor& b);
matcha::Tensor& operator+=(matcha::Tensor& a, const matcha::Tensor& b);

matcha::Tensor operator-(const matcha::Tensor& a, const matcha::Tensor& b);
matcha::Tensor& operator-=(matcha::Tensor& a, const matcha::Tensor& b);
matcha::Tensor operator-(const matcha::Tensor& a);

matcha::Tensor operator*(const matcha::Tensor& a, const matcha::Tensor& b);
matcha::Tensor& operator*=(matcha::Tensor& a, const matcha::Tensor& b);

matcha::Tensor operator/(const matcha::Tensor& a, const matcha::Tensor& b);
matcha::Tensor& operator/=(matcha::Tensor& a, const matcha::Tensor& b);



namespace matcha::engine::fn {


class Add : public ElementwiseBinary {
  public:
    Add(Tensor* a, Tensor* b);
    void run() override;
};

class Subtract : public ElementwiseBinary {
  public:
    Subtract(Tensor* a, Tensor* b);
    void run() override;
};

class Multiply : public ElementwiseBinary {
  public:
    Multiply(Tensor* a, Tensor* b);
    void run() override;
};

class Divide : public ElementwiseBinary {
  public:
    Divide(Tensor* a, Tensor* b);
    void run() override;
};

class Abs : public ElementwiseUnary {
  public:
    explicit Abs(Tensor* a);
    void run() override;
};


}
