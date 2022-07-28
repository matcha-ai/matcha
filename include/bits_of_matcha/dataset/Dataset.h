#pragma once

#include "bits_of_matcha/dataset/Instance.h"
#include "bits_of_matcha/dataset/InstanceIterator.h"
#include "bits_of_matcha/macros/dataset.h"

#include <functional>


namespace matcha {

class Dataset {
public:
  Dataset();

  Instance get() const;
  size_t size() const;
  size_t tell() const;
  void seek(size_t pos) const;
  void reset() const;
  bool eof() const;

  InstanceIterator begin() const;
  InstanceIterator end() const;

  void save(const std::string& file);

public:
  Dataset take(size_t limit) const;
  Dataset batch(size_t limit) const;
  Dataset map(const std::function<Instance (const Instance&)>& function) const;
  Dataset cat(const Dataset& ds) const;

public:
  Dataset(std::initializer_list<tensor> tensors);
  Dataset(const std::vector<tensor>& tensors);
  Dataset(const std::function<Instance ()>& generator);
  Dataset(const std::function<Instance (size_t)>& generator);

public:
  Dataset(const Dataset& other);
  Dataset(Dataset&& other);
  Dataset& operator=(const Dataset& other);
  Dataset& operator=(Dataset&& other);

private:
  friend class Engine;
  explicit Dataset(void* internal);
  void* internal_;
};

}