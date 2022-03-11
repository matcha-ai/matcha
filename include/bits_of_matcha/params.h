#pragma once

#include "bits_of_matcha/frame.h"


namespace matcha {


class Params {
  public:
    Params();
    Params(const Dtype& dtype, const Shape& shape);

    Params& operator=(const Tensor& tensor);

    bool initialized() const;

};


}