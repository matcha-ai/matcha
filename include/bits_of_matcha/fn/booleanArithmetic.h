#pragma once

#include "bits_of_matcha/engine/elementwiseUnary.h"
#include "bits_of_matcha/engine/elementwiseBinary.h"


namespace matcha::fn {

Tensor lnot(const Tensor& a);
Tensor land(const Tensor& a, const Tensor& b);
Tensor lor(const Tensor& a, const Tensor& b);

}

matcha::Tensor operator!(const matcha::Tensor& a);
matcha::Tensor operator&&(const matcha::Tensor& a, const matcha::Tensor& b);
matcha::Tensor operator||(const matcha::Tensor& a, const matcha::Tensor& b);



namespace matcha::engine::fn {


class Lnot : public ElementwiseUnary {
  public:
    Lnot(Tensor* a);
    void run() override;
};

class Land : public ElementwiseBinary {
  public:
    Land(Tensor* a, Tensor* b);
    void run() override;
};

class Lor : public ElementwiseBinary {
  public:
    Lor(Tensor* a, Tensor* b);
    void run() override;
};


}
