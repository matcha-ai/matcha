#pragma once

#include "bits_of_matcha/engine/ElementwiseBinary.h"
#include "bits_of_matcha/engine/ElementwiseUnary.h"


namespace matcha::fn {

tensor square(const tensor& a);
tensor sqrt(const tensor& a);
tensor pow(const tensor& a, const tensor& b);
tensor nrt(const tensor& a, const tensor& b);
tensor exp(const tensor& a);
tensor log(const tensor& a);

}


namespace matcha::engine::fn {

class Square : public ElementwiseUnary {
public:
  Square(Tensor* a);
  void run() override;
};

class Sqrt : public ElementwiseUnary {
public:
  Sqrt(Tensor* a);
  void run() override;
};

class Pow : public ElementwiseBinary {
public:
  Pow(Tensor* a, Tensor* b);
  void run() override;
};

class Exp : public ElementwiseUnary {
public:
  Exp(Tensor* a);
  void run() override;
};

class Log : public ElementwiseUnary {
public:
  Log(Tensor* a);
  void run() override;
};

}