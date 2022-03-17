#pragma once

#include "bits_of_matcha/engine/Fold.h"


namespace matcha::fn {

tensor sum_across(const tensor& a);
tensor product_across(const tensor& a);
tensor sum(const tensor& a);
tensor product(const tensor& a);

tensor max_across(const tensor& a);
tensor min_across(const tensor& a);
tensor max(const tensor& a);
tensor min(const tensor& a);

tensor argmax_across(const tensor& a);
tensor argmin_across(const tensor& a);
tensor argmax(const tensor& a);
tensor argmin(const tensor& a);

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