#pragma once

#include "bits_of_matcha/engine/ElementwiseBinary.h"


namespace matcha::fn {

tensor eq(const tensor& a, const tensor& b);
tensor ne(const tensor& a, const tensor& b);
tensor lt(const tensor& a, const tensor& b);
tensor gt(const tensor& a, const tensor& b);
tensor le(const tensor& a, const tensor& b);
tensor ge(const tensor& a, const tensor& b);

tensor max_between(const tensor& a, const tensor& b);
tensor min_between(const tensor& a, const tensor& b);
tensor max(const tensor& a, const tensor& b);
tensor min(const tensor& a, const tensor& b);

}


matcha::tensor operator==(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator!=(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator>(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator<(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator>=(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator<=(const matcha::tensor& a, const matcha::tensor& b);


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