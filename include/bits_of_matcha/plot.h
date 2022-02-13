#pragma once

#include "bits_of_matcha/tensor.h"

#include <iostream>


namespace matcha {


class Plot {
  public:
    Plot(const Tensor& tensor);

    std::string string() const;

  private:
    Tensor tensor_;

  private:
    std::string asciiHistogram() const;
};


std::ostream& operator<<(std::ostream& os, const Plot& plt);


}
