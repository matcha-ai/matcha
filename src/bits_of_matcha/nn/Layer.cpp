#include "bits_of_matcha/nn/Layer.h"


namespace matcha::nn {

Layer::Layer()
  : initialized_(false)
{}

tensor Layer::operator()(const tensor& a) {
  if (!initialized_) {
    initialized_ = true;
    init(a);
  }
  return run(a);
}

void Layer::init(const tensor& a) {}

}