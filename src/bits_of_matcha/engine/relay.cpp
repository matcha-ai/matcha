#include "bits_of_matcha/engine/relay.h"


namespace matcha {
namespace engine {


Relay::Relay(std::initializer_list<Stream*> sources) {
  sources_.reserve(sources.size());
  for (auto* source: sources) {
    sources_.push_back(new matcha::Stream(matcha::Stream::fromObject(source)));
  }
}

void Relay::eval(Out* out) {
}

void Relay::close(Out* out) {
}

Stream* Relay::source(int idx) {
  return deref(sources_[idx]);
}

const Stream* Relay::source(int idx) const {
  return deref(sources_[idx]);
}

size_t Relay::sources() const {
  return sources_.size();
}


}
}
