#include "bits_of_matcha/View.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/tensor/Binding.h"


namespace matcha {

using Internal = engine::View*;

const Frame& View::frame() const {
  auto&& internal = (Internal) internal_;
  return internal->frame();
}

View::operator tensor() const {
  auto&& internal = (Internal) internal_;
  return ref(internal->read());
}

View& View::operator=(const tensor& t) {
  auto&& internal = (Internal) internal_;
  internal->write(engine::deref(t));
  return *this;
}

auto View::operator[](const tensor& idx) -> View {
  return ref(new engine::View(engine::deref(this), engine::deref(idx)));
}

View::View(void* internal)
  : internal_(internal)
{
  ((Internal) internal_)->ref();
}

View::~View() {
  ((Internal) internal_)->unref();
}

}
