#include "bits_of_matcha/fn/batch.h"
#include "bits_of_matcha/stream.h"

#include <matcha/device>
#include <algorithm>


namespace matcha {
namespace fn {

Stream batch(Stream& stream, size_t sizeLimit) {
  return Stream::fromObject(new engine::fn::Batch(stream, sizeLimit));
}

}

namespace engine {
namespace fn {

Batch::Batch(matcha::Stream& source, size_t sizeLimit)
  : Batch(deref(source), sizeLimit)
{}

Batch::Batch(Stream* source, size_t sizeLimit)
  : Relay{source}
  , ref_{matcha::Stream::fromObject(source)}
  , source_{source}
  , sizeLimit_{sizeLimit}
  , limitReached_{sizeLimit == 0}
  , pos_{0}
  , begin_{source->tell()}
{
}

bool Batch::next() {
  if (eof()) return false;

  source(0)->next();

  pos_++;
  if (pos_ == sizeLimit_) {
    limitReached_ = true;
  }

  return true;
}

bool Batch::seek(size_t pos) {
  pos_ = pos;

  if (pos < sizeLimit_) {
    limitReached_ = false;
    return source(0)->seek(begin_ + pos);
  } else {
    return false;
  }
}

size_t Batch::tell() const {
  return pos_;
}

size_t Batch::size() const {
  return std::min(source(0)->size() - begin_, sizeLimit_);
}

Tensor* Batch::open(int idx) {
  return source(0)->open(idx);
}

void Batch::open(int idx, Tensor* tensor) {
  source(0)->open(idx, tensor);
}

void Batch::relayData(Tensor* relay, Tensor* target) {
}

void Batch::close(Out* out) {
  source(0)->close(out);
}

bool Batch::eof() const {
  if (limitReached_) return true;
  return source(0)->eof();
}


}
}
}
