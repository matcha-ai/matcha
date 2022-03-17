#pragma once

#include "bits_of_matcha/engine/ElementwiseBinary.h"
#include "bits_of_matcha/engine/ElementwiseUnary.h"


namespace matcha::fn {

tensor add(const tensor& a, const tensor& b);
tensor subtract(const tensor& a, const tensor& b);
tensor multiply(const tensor& a, const tensor& b);
tensor divide(const tensor& a, const tensor& b);
tensor negative(const tensor& a);
tensor abs(const tensor& a);

}


matcha::tensor operator+(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor& operator+=(matcha::tensor& a, const matcha::tensor& b);

matcha::tensor operator-(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor& operator-=(matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator-(const matcha::tensor& a);

matcha::tensor operator*(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor& operator*=(matcha::tensor& a, const matcha::tensor& b);

matcha::tensor operator/(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor& operator/=(matcha::tensor& a, const matcha::tensor& b);



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
