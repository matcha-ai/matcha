#include "bits_of_matcha/engine/stream.h"
#include "bits_of_matcha/engine/nodeloader.h"
#include "bits_of_matcha/device/buffer.h"
#include "bits_of_matcha/stream.h"

#include <matcha/engine>
#include <matcha/device>
#include <iostream>


namespace matcha {
namespace engine {

Stream::Stream()
  : Node{}
{
  status_ = {
    .data   = true,
    .update = true,
    .ready  = true
  };
}

Stream* Stream::deref(const matcha::Stream* stream) {
  return stream->object();
}

Stream* Stream::deref(const matcha::Stream& stream) {
  return stream.object();
}

void Stream::prune(Out* link) {
  if (link != nullptr) {
    close(link);
  }
  if (referenced()) return;
  if (outs() > 0) return;
  delete this;
}

void Stream::beginOut(Tensor* tensor) {
  auto* out = createOut(tensor->dtype(), tensor->shape());
  auto* buffer = device::Cpu().createBuffer(tensor->dtype(), tensor->shape());
  buffer->prepare();
  out->setBuffer(buffer);

  outs_.push_back(out);
  tensor->subst(out);
}

}
}
