#include "bits_of_matcha/Slice.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/fn/basic_arithmetic.h"
#include "bits_of_matcha/fn/slice.h"


namespace matcha {

Slice::Slice(tensor* t, const Shape::Range& range)
  : tensor_{t}
  , slice_{range}
{}

Slice::Slice(const Slice* leading, const Shape::Range& range)
  : tensor_{leading->tensor_}
  , slice_(leading->slice_, range)
{}

Slice Slice::operator[](const Shape::Range& range) {
  return Slice(this, range);
}

tensor Slice::get() const {
  return fn::slice(*tensor_, slice_);
}

Slice& Slice::set(const tensor& a) {
  auto& t = *tensor_;
  t = fn::superimpose(t, a, slice_);
  return *this;
}


Slice::operator tensor() const {
  return get();
}

Slice& Slice::operator=(const tensor& a) {
  return set(a);
}

Slice& Slice::operator+=(const tensor& a) {
  return set(*this + a);
}

Slice& Slice::operator-=(const tensor& a) {
  return set(*this - a);
}

Slice& Slice::operator*=(const tensor& a) {
  return set(*this * a);
}

Slice& Slice::operator/=(const tensor& a) {
  return set(*this / a);
}


}