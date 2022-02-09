#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/dtype.h"
#include "bits_of_matcha/shape.h"


namespace matcha {

Tuple::Tuple(std::initializer_list<Tensor> tensors)
  : tensors_(tensors)
{}

Tuple::Tuple(const std::vector<Tensor>& tensors)
  : tensors_{tensors}
{}

const Tensor& Tuple::operator[](int index) const {
  if (index < 0) index += size();
  if (index < 0 || index >= size()) throw std::out_of_range("tuple index is out of range");
  return tensors_[index];
}

size_t Tuple::size() const {
  return tensors_.size();
}

const Tensor* Tuple::begin() const {
  return &tensors_[0];
}

const Tensor* Tuple::end() const {
  return begin() + size();
}

std::ostream& operator<<(std::ostream& os, const Tuple& tuple) {
  os << "Tuple { ";
  for (int i = 0; i < tuple.size(); i++) {
    auto& tensor = tuple[i];
    if (i != 0) os << ", ";
    os << tensor.dtype() << tensor.shape();
  }
  os << " }" << std::endl;
  return os;
}

}
