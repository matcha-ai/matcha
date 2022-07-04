#include "bits_of_matcha/nn/Layer.h"
#include "bits_of_matcha/nn/Net.h"


namespace matcha::nn {

Layer::Layer()
  : initialized_(false)
{}

tensor Layer::operator()(const tensor& a) {
  if (!initialized_) {
    initialized_ = true;
    init(a);
  }
  if (training()) {
    net()->params.add(params);
  }
  return run(a);
}

void Layer::init(const tensor& a) {}

bool Layer::training() const {
  return !netStack_.empty();
}

thread_local std::stack<Net*> Layer::netStack_ {};

Net* Layer::net() {
  if (netStack_.empty()) return nullptr;
  return netStack_.top();
}

}