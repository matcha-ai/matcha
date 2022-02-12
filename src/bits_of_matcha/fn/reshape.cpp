#include "bits_of_matcha/fn/reshape.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor reshape(const Tensor& a, const Shape& shape) {
  auto* node = new engine::fn::Reshape(a, shape);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}


namespace matcha {
namespace engine {
namespace fn {


Reshape::Reshape(const matcha::Tensor& a, const Shape& shape)
  : Reshape(deref(a), shape)
{}

Reshape::Reshape(Tensor* a, const Shape& shape)
  : Node{a}
{
  if (in(0)->size() != shape.size()) {
    throw std::invalid_argument("shapes don't have matching sizes");
  }

  auto* out = createOut(a->dtype(), shape);
  auto* buffer = Context::device().createBuffer(in(0)->dtype(), shape);
  buffer->setSource(in(0)->buffer());
  out->setBuffer(buffer);
  outs_.push_back(out);

  status_ = {
    .data   = in(0)->status().data,
    .update = true,
    .ready  = false
  };
}

void Reshape::dataStatusChanged(In* in) {
  out(0)->dataStatusChanged();

  status_.data = in->status().data;
}

void Reshape::updateStatusChanged(In* in) {
  if (status_.update) return;
  status_.update = true;

  out(0)->updateStatusChanged();
}

void Reshape::bufferChanged(In* in) {
  out(0)->setBuffer(in->buffer());
}

void Reshape::eval(Out* target) {
  if (!status_.update) return;

  status_.update = false;

  in(0)->eval();

  if (!status_.ready) {
    status_.ready = true;
    out(0)->buffer()->prepare();
  }
}

void Reshape::prune(Out* link) {
  if (referenced()) return;
  if (out(0)->linked()) return;
  delete in(0);
  delete this;
}

}
}
}
