#pragma once

#include "bits_of_matcha/Shape.h"


namespace matcha {

class tensor;

class Slice {
public:
  Slice operator[](const Shape::Range& range);

  tensor get() const;
  Slice& set(const tensor& a);

  Slice& operator=(const tensor& a);
  Slice& operator+=(const tensor& a);
  Slice& operator-=(const tensor& a);
  Slice& operator*=(const tensor& a);
  Slice& operator/=(const tensor& a);
  operator tensor() const;

private:
  Slice(const Slice* leading, const Shape::Range& range);
  Slice(tensor* t, const Shape::Range& range);
  tensor* tensor_;
  Shape::Slice slice_;

  friend class tensor;
};


}