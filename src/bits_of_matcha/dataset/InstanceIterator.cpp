#include "bits_of_matcha/dataset/InstanceIterator.h"
#include "bits_of_matcha/engine/dataset/Dataset.h"
#include "bits_of_matcha/print.h"


using Internal = matcha::engine::Dataset*;

namespace matcha {

Instance InstanceIterator::operator*() {
  auto ds = (Internal) internal_;
  ds->seek(pos_);
  return ds->get();
}

void InstanceIterator::operator++() {
  pos_++;
}

bool InstanceIterator::operator==(const InstanceIterator& other) const {
  return pos_ == other.pos_ && internal_ == other.internal_;
}

bool InstanceIterator::operator!=(const InstanceIterator& other) const {
  return !operator==(other);
}

InstanceIterator::InstanceIterator(void* internal, size_t pos)
  : internal_(internal)
  , pos_(pos)
{}

}