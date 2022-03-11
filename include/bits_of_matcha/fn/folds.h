#pragma once

#include "bits_of_matcha/engine/fold.h"


namespace matcha::fn {

Tensor sumAcross(const Tensor& a);
Tensor productAcross(const Tensor& a);
Tensor sum(const Tensor& a);
Tensor product(const Tensor& a);

Tensor maxAcross(const Tensor& a);
Tensor minAcross(const Tensor& a);
Tensor max(const Tensor& a);
Tensor min(const Tensor& a);

Tensor argmaxAcross(const Tensor& a);
Tensor argminAcross(const Tensor& a);
Tensor argmax(const Tensor& a);
Tensor argmin(const Tensor& a);

}


namespace matcha::engine::fn {

class SumAcross : public Fold {
  public:
    SumAcross(Tensor* a);
    void run() override;
};

class ProductAcross : public Fold {
  public:
    ProductAcross(Tensor* a);
    void run() override;
};

class MaxAcross : public Fold {
  public:
    MaxAcross(Tensor* a);
    void run() override;
};

class MinAcross : public Fold {
  public:
    MinAcross(Tensor* a);
    void run() override;
};

class ArgmaxAcross : public Fold {
  public:
    ArgmaxAcross(Tensor* a);
    void run() override;
};

class ArgminAcross : public Fold {
  public:
    ArgminAcross(Tensor* a);
    void run() override;
};

};