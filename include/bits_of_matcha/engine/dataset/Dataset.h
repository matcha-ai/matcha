#pragma once

#include "bits_of_matcha/dataset/Instance.h"
#include "bits_of_matcha/dataset/InstanceIterator.h"

namespace matcha {
class Dataset;
}

namespace matcha::engine {

class Dataset {
public:
  Dataset();

  virtual Instance get() = 0;
  virtual size_t size() const = 0;
  virtual size_t tell() const = 0;
  virtual void seek(size_t pos) = 0;
  virtual ~Dataset();

  InstanceIterator begin();
  InstanceIterator end();

  void ref();
  void unref();
  unsigned refs();

private:
  unsigned refs_;
};

matcha::Dataset ref(Dataset* internal);
Dataset* deref(const matcha::Dataset& external);
Dataset* deref(const matcha::Dataset* external);
Dataset* unref(matcha::Dataset& external);
Dataset* unref(matcha::Dataset* external);

}