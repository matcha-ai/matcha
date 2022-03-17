#pragma once

#include "bits_of_matcha/data/Instance.h"
#include "bits_of_matcha/macros/dataset.h"

#include <cstddef>


namespace matcha {

class Batches;

class Dataset {
public:
  class Internal;
  class Iterator {
  public:
    explicit Iterator(Internal* internal, size_t pos = TELL_POS);
    Instance operator*();

    Iterator& operator++();
    Iterator& operator=(const Iterator& iter);
    bool operator!=(const Iterator& iter) const;

    enum {
      EOF_POS = -1,
      TELL_POS = -2,
    };

  private:
    Internal* internal_;
    size_t pos_;
 };

  Iterator begin() const;
  Iterator end() const;
  Batches batches(size_t batch_size) const;

  size_t size() const;

  class Internal {
  public:
    virtual void seek(size_t pos) = 0;
    virtual size_t tell() const = 0;
    virtual size_t size() const = 0;
    virtual Instance get() = 0;

    bool eof() const;
  };

  explicit Dataset(Internal* internal);

private:
  Internal *internal_;
};

}