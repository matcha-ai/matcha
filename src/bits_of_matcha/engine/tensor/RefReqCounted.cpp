#include "bits_of_matcha/engine/tensor/RefReqCounted.h"

#include <stdexcept>


namespace matcha::engine {

RefReqCounted::RefReqCounted()
  : refs_(0)
  , reqs_(0)
{}

RefReqCounted::~RefReqCounted() {

}

void RefReqCounted::ref() {
  refs_++;
}

void RefReqCounted::req() {
  reqs_++;
}

void RefReqCounted::unref() {
  if (!refs_) throw std::runtime_error("refs are already 0");
  refs_--;
  if (!refs_ && !reqs_) delete this;
}

void RefReqCounted::unreq() {
  if (!reqs_) throw std::runtime_error("reqs are already 0");
  reqs_--;
  if (!refs_ && !reqs_) delete this;
}

unsigned RefReqCounted::refs() const {
  return refs_;
}

unsigned RefReqCounted::reqs() const {
  return reqs_;
}


}