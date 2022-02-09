#include "bits_of_matcha/engine/stream.h"
#include "bits_of_matcha/engine/nodeloader.h"
#include "bits_of_matcha/stream.h"


namespace matcha {
namespace engine {

Stream::Stream()
  : Node{}
{}

Stream* Stream::deref(const matcha::Stream* stream) {
  return stream->object();
}

Stream* Stream::deref(const matcha::Stream& stream) {
  return stream.object();
}

void Stream::require() {

}

void Stream::considerPruning() {

  if (referenced()) return;
  for (auto* in: ins_) in->unbindOut(this);
}

}
}
