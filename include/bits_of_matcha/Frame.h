#pragma once

#include "bits_of_matcha/Dtype.h"
#include "bits_of_matcha/Shape.h"


namespace matcha {

class Frame {
public:
  Frame();
  Frame(Dtype dtype, Shape shape);

  bool null() const;
  const Dtype& dtype() const;
  const Shape& shape() const;

  size_t bytes() const;

  bool operator==(const Frame& frame) const;
  bool operator!=(const Frame& frame) const;

  std::string string() const;

private:
  bool null_;
  Dtype dtype_;
  Shape shape_;

private:
  void assertFrame() const;

};

std::ostream& operator<<(std::ostream& os, const Frame& frame);

}