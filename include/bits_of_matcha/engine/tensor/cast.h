#pragma once

#include "bits_of_matcha/Dtype.h"


namespace matcha::engine {

class Tensor;

Dtype promoteDtypes(Dtype a, Dtype b);
Dtype promoteDtypes(Tensor* a, Tensor* b);

}