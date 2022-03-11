#pragma once

#include "bits_of_matcha/tensor.h"


namespace matcha {

class Instance {
  public:
    Tensor operator[](const std::string& key);

};

class DsIterator {
  public:
    Instance& operator*();

    bool operator!=(const DsIterator& dsIterator);

    Instance& operator++();
};

class Dataset;

class BtIterator {
  public:
    Dataset operator*();
    BtIterator& operator++();
    bool operator!=(const BtIterator& it);

};

class BtIteration {
  public:
    BtIterator begin();

    BtIterator end();
};

class Dataset {
  public:

    DsIterator begin();

    DsIterator end();

    BtIteration batches(size_t size);

    size_t size() const;


};

}


namespace matcha::dataset {

struct Csv {
  operator Dataset();

  std::string file;
  std::vector<std::string> yCols;
};

}