#include "bits_of_matcha/shape.h"

#include <stdexcept>
#include <numeric>


namespace matcha {


Shape::Shape(std::initializer_list<unsigned> dims)
  : dims_{dims}
{}

Shape::Shape(const std::vector<unsigned>& dims)
  : dims_{dims}
{}

size_t Shape::rank() const {
  return dims_.size();
}

size_t Shape::size() const {
  return std::accumulate(
    std::begin(dims_), std::end(dims_),
    1,
    std::multiplies()
  );
}

unsigned Shape::operator[](int index) const {
  if (index < 0) index += rank();
  if (index < 0 || index >= rank()) throw std::out_of_range("axis index is out of range");
  return dims_[index];
}

const unsigned* Shape::begin() const {
  return &dims_[0];
}

const unsigned* Shape::end() const {
  return begin() + rank();
}

bool Shape::operator==(const Shape& other) const {
  if (rank() != other.rank()) return false;
  return std::equal(begin(), end(), other.begin());
}

bool Shape::operator!=(const Shape& other) const {
  return !operator==(other);
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
  os << "[";
  for (int i = 0; i < shape.rank(); i++) {
    if (i != 0) os << ", ";
    os << shape[i];
  }
  os << "]";
  return os;
}

Shape::Reshape::Reshape(const Shape& target)
  : target_(target.begin(), target.end())
{
  check();
}

Shape::Reshape::Reshape(const std::vector<int>& target)
  : target_{target}
{
  check();
}

Shape::Reshape::Reshape(std::initializer_list<int> target)
  : target_{target}
{
  check();
}

void Shape::Reshape::check() {
  bool minusOne = false;
  for (int dim: target_) {
    if (dim == -1) {
      if (minusOne) {
        throw std::invalid_argument("Reshape can deduce at most one dimension (at most -1)");
      }
      minusOne = true;
      continue;
    }
    if (dim == 0) {
      throw std::invalid_argument("Shape dims must be positive");
    }
    if (dim < -1) {
      throw std::invalid_argument("Shape can't contain negative dim/s");
    }
  }
}

Shape Shape::Reshape::operator()(const Shape& shape) const {
  size_t size = 1;
  unsigned deduced = 0;

  for (auto i: target_) {
    if (i == -1) {
      deduced = 1;
    } else {
      size *= i;
    }
  }

  if (shape.size() != size && (shape.size() % size != 0 || !deduced)) {
    throw std::invalid_argument("reshaping failed; shapes are incompatible");
  }

  deduced = shape.size() / size;

  std::vector<unsigned> dims;
  dims.reserve(target_.size());

  for (auto i: target_) {
    if (i == -1) {
      dims.push_back(deduced);
    } else {
      dims.push_back(i);
    }
  }

  return dims;
}


}
