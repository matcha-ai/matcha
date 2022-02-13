#pragma once

#include "bits_of_matcha/tensor.h"

#include <iostream>


namespace matcha {


class Plt {
  public:
    Plt(const Tensor& tensor);

    std::string string() const;

  private:
    Tensor tensor_;

  private:
    std::string asciiHistogram() const;
};


std::ostream& operator<<(std::ostream& os, const Plt& plt);


}
