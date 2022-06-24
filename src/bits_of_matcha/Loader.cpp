#include "bits_of_matcha/Loader.h"
#include "matcha/dataset"
#include "matcha/tensor"
#include "bits_of_matcha/engine/ops/LoadCsv.h"

#include <filesystem>


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
  std::filesystem::path path(file_);
  std::string ext = path.extension();
  std::transform(ext.begin(), ext.end(), ext.begin(), tolower);
  if (ext == ".csv") {
    auto op = new engine::ops::LoadCsv(file_);
    auto out = engine::ref(op->outputs[0]);
    engine::send(op);
    return out;
  } else {
    throw std::runtime_error("unsupported load format: " + ext);
  }
}

Loader::operator Flow() {
  return {};
}

}