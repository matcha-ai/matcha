#include "bits_of_matcha/data/Instance.h"


namespace matcha {

tensor Instance::operator[](const std::string& key) {
  return data_.at(key);
}

std::vector<std::string> Instance::keys() {
  std::vector<std::string> result;
  result.reserve(data_.size());
  for (auto& [key, val]: data_) {
    result.push_back(key);
  }
  return result;
}

Instance::Instance(std::map<std::string, tensor> data)
  : data_{std::move(data)}
{}

}