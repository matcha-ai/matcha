#pragma once

#include "bits_of_matcha/dtype.h"
#include "bits_of_matcha/shape.h"


namespace matcha {

class Frame {
  public:
    Frame();
    Frame(Dtype dtype, Shape shape);

    bool null() const;
    const Dtype* dtype() const;
    const Shape* shape() const;

    size_t bytes() const;

  private:
    bool null_;
    Dtype dtype_;
    Shape shape_;

  private:
    void assertFrame() const;

};

}