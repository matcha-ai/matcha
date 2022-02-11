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
  : ref_{matcha::Stream::fromObject(source)}
  , source_{source}
  , sizeLimit_{sizeLimit}
  , limitReached_{sizeLimit == 0}
{
}

void Batch::reset() {

}

void Batch::shuffle() {

}

bool Batch::eof() const {
  if (limitReached_) return true;
  return source_->eof();
}

size_t Batch::size() const {
  return std::min(source_->size(), sizeLimit_);
}

Tensor* Batch::open() {
  auto* relay = source_->open();
  auto* out = new Tensor(relay->dtype(), relay->shape());
  relayData(relay, out);
  return out;
}

void Batch::open(Tensor* out) {
  auto* relay = new Tensor(out->dtype(), out->shape());
  source_->open(relay);
  relayData(relay, out);
}

void Batch::relayData(Tensor* relay, Tensor* target) {
  auto* out = createOut(target->dtype(), target->shape()) ;
  out->setBuffer(relay->buffer());

  relays_.push_back(relay);
  outs_.push_back(out);

  target->subst(out);

  relay->rename("batchRelay");
}

void Batch::close(Out* out) {
  std::cout << "close Batch out" << std::endl;
  auto* relay = relays_[out->id()];
  relay->prune();
  outs_.erase(std::remove(std::begin(outs_), std::end(outs_), out));
  relays_.erase(std::remove(std::begin(relays_), std::end(relays_), relay));
}

void Batch::eval(Out* out) {
  unsigned id = out->id();
  if (id >= 64) throw std::runtime_error("Batch out id overflow");
  position_.resize(id + 1);

  auto& pos = position_[id];
  if (pos < sizeLimit_) {
    if (++pos == sizeLimit_) limitReached_ = true;
  } else {
    throw std::out_of_range("Batch limit reached");
  }

  auto* relay = relays_[id];
  relay->updateStatusChanged();
  relay->eval();
}


}
}
}
