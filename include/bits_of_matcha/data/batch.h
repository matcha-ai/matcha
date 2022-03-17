#pragma once

#include "bits_of_matcha/data/Dataset.h"


namespace matcha {

class Batches {
public:
  class Iterator {
  public:
    Dataset& operator*();
    const Dataset& operator*() const;

    Iterator& operator++();
    Iterator& operator=(const Iterator& iter);
    bool operator!=(const Iterator& iter);
  };

  Iterator begin();
  Iterator end();
};

}