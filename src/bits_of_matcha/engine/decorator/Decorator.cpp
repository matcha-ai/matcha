#include "bits_of_matcha/engine/decorator/Decorator.h"


namespace matcha::engine {

Decorator::Decorator(const fn& preimage)
  : preimage_(preimage)
{}

Decorator::Decorator()
{}

auto Decorator::preimage() -> fn& { return preimage_; }
auto Decorator::preimage() const -> const fn& { return preimage_; }
bool Decorator::hasPreimage() const { return preimage_; };
void Decorator::setPreimage(const fn& preimage) { preimage_ = preimage; }

std::vector<Tensor*> Decorator::run(const std::vector<Tensor*>& inputs) {
  std::cout << "runnign " << this << std::endl;
  auto results = preimage_(ref(inputs));
  return unref(results);
}

fn ref(std::shared_ptr<Decorator> decorator) {
  auto& variant = decorator->preimage().stdVariant();

  if (std::holds_alternative<UnaryOp>(variant)) {
    return (fn) [decorator] (const tensor& a) {
      return ref(decorator->run({deref(a)})[0]);
    };
  } else if (std::holds_alternative<BinaryOp>(variant)) {
    return (fn) [decorator] (const tensor& a, const tensor& b) {
      return ref(decorator->run({deref(a), deref(b)})[0]);
    };
  } else if (std::holds_alternative<TernaryOp>(variant)) {
    return (fn) [decorator] (const tensor& a, const tensor& b, const tensor& c) {
      return ref(decorator->run({deref(a), deref(b), deref(c)})[0]);
    };
  } else if (std::holds_alternative<NaryOp>(variant)) {
    return (fn) [decorator] (const tuple& inputs) {
      return engine::ref(decorator->run(deref(inputs)));
    };
  } else {
    throw std::bad_variant_access();
  }
}

}