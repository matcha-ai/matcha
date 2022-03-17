#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/Frame.h"


namespace matcha {


Frame::Frame()
  : null_{true}
  , dtype_{Float}
  , shape_{}
{}

Frame::Frame(Dtype dtype, Shape shape)
  : null_{false}
  , dtype_{dtype}
  , shape_{std::move(shape)}
{

}

bool Frame::null() const {
  return null_;
}

const Dtype* Frame::dtype() const {
  assertFrame();
  return &dtype_;
}

const Shape* Frame::shape() const {
  assertFrame();
  return &shape_;
}

void Frame::assertFrame() const {
  if (null_) throw std::runtime_error("Frame is Null");
}

size_t Frame::bytes() const {
  assertFrame();
  return dtype_.size() * shape_.size();
}

}