#include "bits_of_matcha/dataset/Dataset.h"
#include "bits_of_matcha/engine/dataset/Dataset.h"
#include "bits_of_matcha/print.h"

#include "bits_of_matcha/dataset/loaders/Tensors.h"
#include "bits_of_matcha/dataset/loaders/Generator.h"
#include "bits_of_matcha/dataset/loaders/Take.h"
#include "bits_of_matcha/dataset/loaders/Map.h"


using namespace matcha::engine;

namespace matcha {

Dataset::Dataset()
  : Dataset(dataset::Tensors {})
{}

Dataset::Dataset(const Dataset& other)
  : internal_(other.internal_)
{
  if (internal_) deref(this)->ref();
}

Dataset::Dataset(Dataset&& other)
  : internal_(other.internal_)
{
  other.internal_ = nullptr;
}

Dataset& Dataset::operator=(const Dataset& other) {
  if (internal_) deref(this)->unref();
  internal_ = other.internal_;
  if (internal_) deref(this)->ref();
  return *this;
}

Dataset& Dataset::operator=(Dataset&& other) {
  internal_ = other.internal_;
  other.internal_ = nullptr;
  return *this;
}

size_t Dataset::size() const {
  if (!internal_) throw std::runtime_error("dataset is null");
  return deref(this)->size();
}

Instance Dataset::get() const {
  if (!internal_) throw std::runtime_error("dataset is null");
  return deref(this)->get();
}

void Dataset::seek(size_t pos) const {
  if (!internal_) throw std::runtime_error("dataset is null");
  return deref(this)->seek(pos);
}

size_t Dataset::tell() const {
  if (!internal_) throw std::runtime_error("dataset is null");
  return deref(this)->tell();
}

InstanceIterator Dataset::begin() const {
  if (!internal_) throw std::runtime_error("dataset is null");
  return deref(this)->begin();
}

InstanceIterator Dataset::end() const {
  if (!internal_) throw std::runtime_error("dataset is null");
  return deref(this)->end();
}

Dataset Dataset::take(size_t limit) const {
  return dataset::Take(*this, limit);
}

Dataset Dataset::map(const std::function<Instance (const Instance&)>& function) const {
  return dataset::Map(*this, function);
}

Dataset::Dataset(void* internal)
  : internal_(internal)
{
  if (internal_) deref(this)->ref();
}

Dataset::Dataset(const std::vector<tensor>& tensors)
  : Dataset(dataset::Tensors(tensors))
{}

Dataset::Dataset(std::initializer_list<tensor> tensors)
  : Dataset(dataset::Tensors(tensors))
{}

Dataset::Dataset(const std::function<Instance()>& generator)
  : Dataset(dataset::Generator(generator))
{}

Dataset::Dataset(const std::function<Instance (size_t)>& generator)
  : Dataset(dataset::Generator(generator))
{}

}