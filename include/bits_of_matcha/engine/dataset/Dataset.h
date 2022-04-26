#pragma once

#include "bits_of_matcha/dataset/Instance.h"


namespace matcha::engine {

class Dataset {
  virtual Instance read();

  virtual size_t size() const;
  virtual size_t tell() const;
  virtual void seek(size_t pos) const;

};

}