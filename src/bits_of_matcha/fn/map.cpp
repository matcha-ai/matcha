#include "bits_of_matcha/fn/map.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/stream.h"
#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/engine/stream.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Stream map(Stream& stream, const std::function<Tensor (const Tensor&)> fn) {
  return Stream::fromObject(new engine::fn::Map(stream, fn));
}

}

namespace engine {
namespace fn {

Map::Map(matcha::Stream& source, std::function<matcha::Tensor (const matcha::Tensor&)> fn)
  : Map(deref(source), fn)
{

}

Map::Map(Stream* source, std::function<matcha::Tensor (const matcha::Tensor&)> fn)
  : ref_{matcha::Stream::fromObject(source)}
  , source_{source}
  , fn_{fn}
{

}

void Map::reset() {
  source_->reset();
}

void Map::shuffle() {
  source_->shuffle();
}

bool Map::eof() const {
  return source_->eof();
}

size_t Map::size() const {
  return source_->size();
}

Tensor* Map::open() {
  auto* relay  = source_->open();
  auto mapping = fn_(matcha::Tensor::fromObject(relay));
  auto* result = Node::deref(mapping);
  auto* target = new Tensor(result->dtype(), result->shape());
  release(mapping);
  relayData(relay, result, target);
  return target;
}

void Map::open(Tensor* target) {
  auto* relay  = new Tensor(target->dtype(), target->shape());
  source_->open(relay);
  auto mapping = fn_(matcha::Tensor::fromObject(relay));
  auto* result = Node::deref(mapping);
  release(mapping);
  relayData(relay, result, target);
}

void Map::relayData(Tensor* relay, Tensor* result, Tensor* target) {
  auto* out = createOut(result->dtype(), result->shape(), outs_.size());
  result->buffer()->prepare();
  out->setBuffer(result->buffer());

  relays_.push_back(relay);
  results_.push_back(result);
  outs_.push_back(out);

  target->subst(out);

  relay->rename("mapRelay");
  result->rename("mapResult");
}

void Map::close(Out* out) {
  unsigned id = out->id();
  auto* relay = relays_[id];
  auto* result = results_[id];
  result->prune();
  outs_.erase(std::remove(std::begin(outs_), std::end(outs_), out));
  relays_.erase(std::remove(std::begin(relays_), std::end(relays_), relay));
  results_.erase(std::remove(std::begin(results_), std::end(results_), result));
}

void Map::eval(Out* out) {
  unsigned id = out->id();
  auto* relay = relays_[id];
  auto* result = results_[id];

  relay->updateStatusChanged();
  result->eval();
}


}
}
}
