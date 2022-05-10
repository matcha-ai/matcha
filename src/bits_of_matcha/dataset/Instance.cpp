#include "bits_of_matcha/dataset/Instance.h"


namespace matcha {

Instance::Instance(const std::map<std::string, tensor>& data)
  : data_(data)
{}

Instance::Instance(std::initializer_list<std::pair<std::string, tensor>> data)
  : data_(data.begin(), data.end())
{}

Instance::Instance(const std::string& key, const tensor& value)
  : data_{{key, value}}
{}

tensor& Instance::operator[](const std::string& key) {
  return data_[key];
}

const tensor& Instance::operator[](const std::string& key) const {
  try {
    return data_.at(key);
  } catch (std::out_of_range&) {
    throw std::out_of_range("unknown Instance key");
  }
}

size_t Instance::size() const {
  return data_.size();
}

std::vector<std::string> Instance::keys() const {
  std::vector<std::string> k;
  k.reserve(size());
  for (auto&[key, value]: data_) {
    k.push_back(key);
  }
  return k;
}

std::vector<tensor> Instance::values() const {
  std::vector<tensor> v;
  v.reserve(size());
  for (auto&[key, value]: data_) {
    v.push_back(value);
  }
  return v;
}

Instance::operator bool() const {
  return !data_.empty();
}

}

std::ostream& operator<<(std::ostream& os, const matcha::Instance& instance) {
  auto keys = instance.keys();
  auto vals = instance.values();

  for (int i = 0; i < instance.size(); i++) {
    os << keys[i] << "\n" << vals[i] << "\n";
  }
  return os;
}

