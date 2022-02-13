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
  : Relay{source}
  , ref_{matcha::Stream::fromObject(source)}
  , source_{source}
  , fn_{fn}
{}

Tensor* Map::open(int idx) {
  auto* relay  = source(0)->open(idx);
  auto mapping = fn_(matcha::Tensor::fromObject(relay));
  auto* result = Node::deref(mapping);
  release(mapping);
  return result;
}

void Map::open(int idx, Tensor* target) {
  auto* relay  = new Tensor(target->dtype(), target->shape());
  source(0)->open(idx, relay);
  auto mapping = fn_(matcha::Tensor::fromObject(relay));
  auto* result = Node::deref(mapping);
  release(mapping);
  target->subst(result->out());
}

void Map::relayData(Tensor* relay, Tensor* result, Tensor* target) {
}

void Map::close(Out* out) {
}

bool Map::next() {
  return source(0)->next();
}

bool Map::seek(size_t pos) {
  return source(0)->seek(pos);
}

size_t Map::tell() const {
  return source(0)->tell();
}

size_t Map::size() const {
  return source(0)->size();
}

bool Map::eof() const {
  return source(0)->eof();
}


}
}
}
