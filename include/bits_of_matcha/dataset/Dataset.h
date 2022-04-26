#pragma once

#include "bits_of_matcha/dataset/Instance.h"
#include "bits_of_matcha/macros/dataset.h"


namespace matcha {

class Dataset {
public:
  Instance read();

  size_t size() const;
  size_t tell() const;
  void seek(size_t pos) const;

  std::vector<std::string> keys();

private:
  void* internal_;
};

}