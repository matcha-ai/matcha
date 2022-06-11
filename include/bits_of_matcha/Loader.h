#pragma once

#include <string>


namespace matcha {

class Dataset;
class tensor;
class Flow;

class Loader {
public:
  operator Dataset();
  operator tensor();
  operator Flow();

private:
  explicit Loader(const std::string& file);

  std::string file_;

  friend Loader load(const std::string&);
};

Loader load(const std::string& file);

}