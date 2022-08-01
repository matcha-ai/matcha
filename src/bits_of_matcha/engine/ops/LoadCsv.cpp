#include "bits_of_matcha/engine/ops/LoadCsv.h"

#include <fstream>
#include <sstream>


namespace matcha::engine::ops {

LoadCsv::LoadCsv(const std::string& file)
  : Op{}
  , file_(file)
{
  addOutput(getFrame());
}

LoadCsv::LoadCsv(const std::string& file, const Frame& frame)
  : Op{}
  , file_(file)
{
  addOutput(frame);
}

void LoadCsv::run() {
  auto f = outputs[0]->malloc().as<float*>();
  std::ifstream is(file_);

  std::stringstream ss;
  std::string buff, buff2;

  do {
    std::getline(is, buff);
    ss = std::stringstream(buff);
    std::getline(ss, buff2, ',');
  } while (buff2 == "dtype" || buff2 == "shape");

  // skip header row if present

  try {
    ss = std::stringstream(buff);
    while (std::getline(ss, buff2, ','))
      float val = std::stof(buff2);
  } catch (std::invalid_argument&) {
    std::getline(is, buff);
  }

  size_t size = outputs[0]->size();
  float* end = f + size;
  float* it = f;

  do {
    ss = std::stringstream(buff);
    while (std::getline(ss, buff2, ',')) {
      if (it == end)
        throw std::runtime_error("can't reshape the CSV data blob to fit the tensor");
      float val = std::stof(buff2);
      *it++ = val;
    }
  } while (std::getline(is, buff));

  if (it != end)
    throw std::runtime_error("can't reshape the CSV data blob to fit the tensor");
}

Frame LoadCsv::getFrame() {
  Dtype dtype = Float;
  std::vector<unsigned> dims;

  std::ifstream is(file_);
  std::string buff;

  std::stringstream ss;
  std::string buff2;

  std::getline(is, buff);
  ss = std::stringstream(buff);
  std::getline(ss, buff2, ',');

  if (buff2 == "dtype") {
    std::getline(ss, buff2, ',');
    dtype = Float;

    std::getline(ss, buff2, ',');
    if (ss) {
      throw std::runtime_error("csv parse failed, expected newline after dtype");
    }



    std::getline(is, buff);
    ss = std::stringstream(buff);
    std::getline(ss, buff2, ',');
  }

  if (buff2 == "shape") {
    while (std::getline(ss, buff2, ',')) {
      unsigned dim = std::stoul(buff2);
      dims.push_back(dim);
    }
  } else {
    unsigned cols = 0;
    bool header = false;
    ss = std::stringstream(buff);
    while (std::getline(ss, buff2, ',')) {
      try {
        stof(buff2);
      } catch (std::invalid_argument&) {
        header = true;
      }
      cols++;
    }

    unsigned rows = header ? 0 : 1;
    while(std::getline(is, buff)) {
      if (buff.empty()) continue;
      rows++;
    }

    dims = {rows, cols};
  }

  return {dtype, dims};
}

}
