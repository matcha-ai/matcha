#pragma once

#include "bits_of_matcha/object.h"

#include <initializer_list>
#include <vector>
#include <iostream>


namespace matcha {

class Tensor;

class Tuple : Object {
  public:
    Tuple(std::initializer_list<Tensor> tensors);
    Tuple(const std::vector<Tensor>& tensors);

    const Tensor& operator[](int index) const;
    size_t size() const;

    const Tensor* begin() const;
    const Tensor* end() const;

  private:
    std::vector<Tensor> tensors_;

    friend std::ostream& operator<<(std::ostream& os, const Tuple& tuple);
};

std::ostream& operator<<(std::ostream& os, const Tuple& tuple);

}
