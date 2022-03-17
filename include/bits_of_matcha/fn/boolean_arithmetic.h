#pragma once

#include "bits_of_matcha/engine/ElementwiseUnary.h"
#include "bits_of_matcha/engine/ElementwiseBinary.h"


namespace matcha::fn {

tensor lnot(const tensor& a);
tensor land(const tensor& a, const tensor& b);
tensor lor(const tensor& a, const tensor& b);

}

matcha::tensor operator!(const matcha::tensor& a);
matcha::tensor operator&&(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator||(const matcha::tensor& a, const matcha::tensor& b);



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
