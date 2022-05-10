#pragma once

#include "bits_of_matcha/dataset/Instance.h"

namespace matcha {

class InstanceIterator {
public:
  Instance operator*();
  void operator++();

  bool operator==(const InstanceIterator& other) const;
  bool operator!=(const InstanceIterator& other) const;

private:
  friend class Engine;
  explicit InstanceIterator(void* internal, size_t pos);
  void* internal_;
  size_t pos_;
};

}