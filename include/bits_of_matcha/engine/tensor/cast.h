#pragma once

#include "bits_of_matcha/Dtype.h"

#include <complex>


namespace matcha::engine {

class Tensor;

Dtype promoteDtypes(Dtype a, Dtype b);
Dtype promoteDtypes(Tensor* a, Tensor* b);

template <class T>
inline bool operator>(std::complex<T> a, std::complex<T> b) {
  return a.real() > b.real();
}

template <class T>
inline bool operator<(std::complex<T> a, std::complex<T> b) {
  return a.real() < b.real();
}

template <class T>
inline bool operator>=(std::complex<T> a, std::complex<T> b) {
  return a.real() >= b.real();
}

template <class T>
inline bool operator<=(std::complex<T> a, std::complex<T> b) {
  return a.real() <= b.real();
}

}