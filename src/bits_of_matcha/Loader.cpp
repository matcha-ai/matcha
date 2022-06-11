#include "bits_of_matcha/Loader.h"
#include "matcha/dataset"
#include "matcha/tensor"


namespace matcha {

Loader::Loader(const std::string& file)
  : file_(file)
{}

Loader load(const std::string& file) {
  return Loader(file);
}

Loader::operator Dataset() {
  return dataset::Csv {file_};
}

Loader::operator tensor() {
  return 0;
}

Loader::operator Flow() {
  return {};
}

}