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

std::string Frame::string() const {
  if (null_) return "NullFrame";

  std::string buffer = dtype_.string();
  buffer += "[";

  for (int i = 0; i < shape_.rank(); i++) {
    if (i != 0) buffer += ", ";
    buffer += std::to_string(shape_[i]);
  }
  buffer += "]";
  return buffer;
}

void Frame::assertFrame() const {
  if (null_) throw std::runtime_error("Frame is Null");
}

size_t Frame::bytes() const {
  assertFrame();
  return dtype_.size() * shape_.size();
}

bool Frame::operator==(const Frame& frame) const {
  return dtype_ == frame.dtype_ && shape_ == frame.shape_;
}

bool Frame::operator!=(const Frame& frame) const {
  return !operator==(frame);
}

}