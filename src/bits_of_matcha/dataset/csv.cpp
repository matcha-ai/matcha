#include "bits_of_matcha/dataset/csv.h"
#include "bits_of_matcha/engine/Tensor.h"
#include "bits_of_matcha/print.h"

#include <sstream>


namespace matcha::dataset {


Csv::Internal::Internal(Csv* csv)
  : pos_{0}
{
  is_.open(csv->file);
  if (!is_.is_open()) {
    throw std::invalid_argument("failed to open csv file");
  }
  size_ = -1;
  readRow(head_);
  findColsXY(csv->y_cols);
  seek(-1);
  size_ = pos_;
  seek(0);
}

void Csv::Internal::readRow(std::vector<std::string>& buffer) {
  buffer.clear();

  std::string rowBuffer;
  std::getline(is_, rowBuffer);
  std::stringstream rowStream(rowBuffer);

  std::string colBuffer;
  while (rowStream.peek() != EOF) {
    std::getline(rowStream, colBuffer, ',');
    buffer.push_back(colBuffer);
  }
}

void Csv::Internal::findColsXY(std::vector<std::string> y) {
  for (auto& col: y) {
    std::transform(
      std::begin(col), std::end(col),
      std::begin(col),
      ::tolower
    );
  }
  for (unsigned i = 0; i < head_.size(); i++) {
    auto col = head_[i];
    std::transform(
      std::begin(col), std::end(col),
      std::begin(col),
      ::tolower
    );

    auto inY = std::find(std::begin(y), std::end(y), col);
    if (inY != std::end(y)) {
      colsY_.push_back(i);
    } else {
      colsX_.push_back(i);
    }
  }
}

Csv::Internal* Csv::init() {
  return new Internal(this);
}

engine::Tensor* Csv::Internal::getX() {
  if (colsX_.empty()) return nullptr;
  auto t = new engine::Tensor(Float, {(unsigned) colsX_.size()});
  auto vals = (float*) t->writeBuffer()->payload();
  int i = 0;
  for (auto col: colsX_) {
    vals[i] = std::stof(rowBuffer_[col]);
    i++;
  }
  return t;
}

engine::Tensor* Csv::Internal::getY() {
  if (colsY_.empty()) return nullptr;
  auto t = new engine::Tensor(Float, {(unsigned) colsY_.size()});
  auto vals = (float*) t->writeBuffer()->payload();
  int i = 0;
  for (auto col: colsY_) {
    vals[i] = std::stof(rowBuffer_[col]);
    i++;
  }
  return t;
}

Instance Csv::Internal::get() {
  if (is_.peek() == EOF) {
    throw std::out_of_range("eof");
  }
  readRow(rowBuffer_);
  pos_++;

  std::map<std::string, tensor> data;
  auto x = getX();
  auto y = getY();
  if (x) {
    data["x"] = tensor(x);
  }
  if (y) {
    data["y"] = tensor(y);
  }
  return Instance(data);
}

size_t Csv::Internal::tell() const {
  return pos_;
}

size_t Csv::Internal::size() const {
  return size_;
}

void Csv::Internal::seek(size_t pos) {
  if (pos < pos_) {
    is_.seekg(0);
    skipRows(1);
    pos_ = 0;
  }
  pos_ += skipRows(pos - pos_);
}

size_t Csv::Internal::skipRows(size_t amount) {
  std::string buffer;
  size_t i;
  for (i = 0; i < amount; i++) {
    if (is_.peek() == EOF) break;
    std::getline(is_, buffer);
  }
  return i;
}


}