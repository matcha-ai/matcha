#pragma once

#include "bits_of_matcha/engine/elementwiseBinary.h"
#include "bits_of_matcha/engine/elementwiseUnary.h"


namespace matcha::fn {

Tensor square(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor pow(const Tensor& a, const Tensor& b);
Tensor nrt(const Tensor& a, const Tensor& b);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);

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