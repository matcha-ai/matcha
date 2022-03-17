#include "bits_of_matcha/data/Dataset.h"
#include "bits_of_matcha/print.h"


namespace matcha {

bool Dataset::Internal::eof() const {
  return tell() == size();
}

Dataset::Iterator::Iterator(Internal* internal, size_t pos)
  : internal_{internal}
{
  if (pos == EOF_POS) {
    pos_ = internal_->size();
  } else if (pos == TELL_POS) {
    pos_ = internal_->tell();
  } else {
    pos_ = pos;
  }
}

Instance Dataset::Iterator::operator*() {
  internal_->seek(pos_);
  return internal_->get();
}

Dataset::Iterator& Dataset::Iterator::operator++() {
  pos_++;
  return *this;
}

bool Dataset::Iterator::operator!=(const Iterator& iter) const {
  return pos_ != iter.pos_;
}

Dataset::Dataset(Internal* internal)
  : internal_{internal}
{

}

Dataset::Iterator Dataset::begin() const {
  return Iterator(internal_, 0);
}

Dataset::Iterator Dataset::end() const {
  return Iterator(internal_, size());
}

size_t Dataset::size() const {
  return internal_->size();
}

}
