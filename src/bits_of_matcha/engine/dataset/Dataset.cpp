#include "bits_of_matcha/engine/dataset/Dataset.h"
#include "bits_of_matcha/dataset/Dataset.h"
#include "bits_of_matcha/Engine.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

Dataset::Dataset()
  : refs_(0)
{}

Dataset::~Dataset() {

}

void Dataset::ref() {
  refs_++;
}

void Dataset::unref() {
  if (refs_ == 0) throw std::runtime_error("dataset refs are already 0");
  refs_--;
  if (refs_ == 0) delete this;
}

InstanceIterator Dataset::begin() {
  return Engine::makeInstanceIterator(this, 0);
}

InstanceIterator Dataset::end() {
  return Engine::makeInstanceIterator(this, size());
}

matcha::Dataset ref(Dataset* internal) {
  return Engine::ref(internal);
}

Dataset* deref(const matcha::Dataset& external) {
  return Engine::deref(external);
}

Dataset* deref(const matcha::Dataset* external) {
  return Engine::deref(external);
}

Dataset* unref(matcha::Dataset& external) {
  return Engine::unref(external);
}

Dataset* unref(matcha::Dataset* external) {
  return Engine::unref(external);
}


}