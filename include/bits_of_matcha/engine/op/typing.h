#pragma once

#include "bits_of_matcha/Dtype.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"

#include <complex>


namespace matcha::engine {

class Tensor;

Dtype promoteDtypes(Dtype a, Dtype b);
Dtype promoteDtypes(Tensor* a, Tensor* b);

bool isReal(Dtype dtype);
bool isComplex(Dtype dtype);

bool isFloating(Dtype dtype);
bool isSigned(Dtype dtype);
bool isUnsigned(Dtype dtype);

bool isFloatingReal(Dtype dtype);
bool isSignedReal(Dtype dtype);
bool isUnsignedReal(Dtype dtype);

bool isFloatingComplex(Dtype dtype);
bool isSignedComplex(Dtype dtype);
bool isUnsignedComplex(Dtype dtype);

inline bool isReal(Tensor* t) { return isReal(t->dtype()); }
inline bool isComplex(Tensor* t) { return isComplex(t->dtype()); }

inline bool isFloating(Tensor* t) { return isFloating(t->dtype()); }
inline bool isSigned(Tensor* t) { return isSigned(t->dtype()); }
inline bool isUnsigned(Tensor* t) { return isUnsigned(t->dtype()); }

inline bool isFloatingReal(Tensor* t) { return isFloatingReal(t->dtype()); }
inline bool isSignedReal(Tensor* t) { return isSignedReal(t->dtype()); }
inline bool isUnsignedReal(Tensor* t) { return isUnsignedReal(t->dtype()); }

inline bool isFloatingComplex(Tensor* t) { return isFloatingComplex(t->dtype()); }
inline bool isSignedComplex(Tensor* t) { return isSignedComplex(t->dtype()); }
inline bool isUnsignedComplex(Tensor* t) { return isUnsignedComplex(t->dtype()); }

}