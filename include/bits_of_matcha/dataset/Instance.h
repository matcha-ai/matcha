#pragma once

#include "bits_of_matcha/tensor.h"

#include <iostream>
#include <map>
#include <string>


namespace matcha {

class Instance {
public:
  Instance(const std::map<std::string, tensor>& data);
  Instance(std::initializer_list<std::pair<std::string, tensor>> data);
  Instance(const std::string& key, const tensor& value);

  tensor& operator[](const std::string& key);
  const tensor& operator[](const std::string& key) const;

  size_t size() const;
  std::vector<std::string> keys() const;
  std::vector<tensor> values() const;

  operator bool() const;

private:
  std::map<std::string, tensor> data_;
};

}

std::ostream& operator<<(std::ostream& os, const matcha::Instance& i);
