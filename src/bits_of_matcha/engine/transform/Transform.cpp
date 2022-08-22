#include "bits_of_matcha/engine/transform/Transform.h"


namespace matcha::engine {

Transform::Transform(const fn& preimage)
  : preimage_(preimage)
{}

Transform::Transform()
{}

auto Transform::preimage() -> fn& { return preimage_; }
auto Transform::preimage() const -> const fn& { return preimage_; }
bool Transform::hasPreimage() const { return preimage_; };
void Transform::setPreimage(const fn& preimage) { preimage_ = preimage; }

std::vector<Tensor*> Transform::run(const std::vector<Tensor*>& inputs) {
  auto results = preimage_(ref(inputs));
  return unref(results);
}

fn ref(std::shared_ptr<Transform> transform) {
  auto& variant = transform->preimage().stdVariant();

  if (std::holds_alternative<UnaryOp>(variant)) {
    return (fn) [transform] (const tensor& a) {
      return ref(transform->run({deref(a)})[0]);
    };
  } else if (std::holds_alternative<BinaryOp>(variant)) {
    return (fn) [transform] (const tensor& a, const tensor& b) {
      return ref(transform->run({deref(a), deref(b)})[0]);
    };
  } else if (std::holds_alternative<TernaryOp>(variant)) {
    return (fn) [transform] (const tensor& a, const tensor& b, const tensor& c) {
      return ref(transform->run({deref(a), deref(b), deref(c)})[0]);
    };
  } else if (std::holds_alternative<NaryOp>(variant)) {
    return (fn) [transform] (const tuple& inputs) {
      return engine::ref(transform->run(deref(inputs)));
    };
  } else {
    throw std::bad_variant_access();
  }
}

}