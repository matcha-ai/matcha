#include "bits_of_matcha/View.h"


namespace matcha {

Dtype View::dtype() const {
  return tensor_->dtype();
}

Shape View::shape() const {
  return {};
}

Frame View::frame() const {
  return {dtype(), shape()};
}

}