#include "bits_of_matcha/engine/ops/Print.h"

#include <iostream>
#include <sstream>


namespace matcha::engine::ops {

Print::Print(Tensor* a, bool endl, std::ostream& os)
  : Op{a}
  , os_(os)
  , endl_(endl)
{
  recorded_ = recorder_.str();
  recorder_.str("");
}

Print::Print(const std::string& text, bool endl, std::ostream& os)
  : Op{}
  , text_(text)
  , endl_(endl)
  , os_(os)
{
  recorded_ = recorder_.str();
  recorder_.str("");
}

OpMeta<Print> Print::meta {
  .name = "Print",
  .sideEffect = true,
};

void Print::run() {
  dumpRecorded(std::cout);

  if (inputs.any())
    dumpTensor(os_);
  else
    dumpText(os_);

  if (endl_)
    std::endl(os_);
}

void Print::dumpRecorded(std::ostream& os) {
  os << recorded_;
}

void Print::dumpText(std::ostream& os) {
  os << text_;
}

void Print::dumpTensor(std::ostream& os) {
//  os << this << " tensorrrr" << std::endl;
//  print(frame_.string());
//  print(buffer());
//  os << this << " ";
//  os << "done";
//  return;
  auto t = inputs[0];

  if (t->frame().null()) {
    os << "NullTensor";
    return;
  }

  if (!t->buffer()) {
    os << t->dtype() << t->shape();
    return;
  }

//  print("buffer is now: ", buffer());
  auto floats = t->buffer()->as<float*>();

  if (t->rank() == 0) {
    os << floats[0];
    return;
  }

  auto iter = MatrixStackIteration(t->shape());
  bool oneline = iter.rows == 1;

  size_t cellW = 0;
  for (size_t i = 0; i < t->size(); i++) {
    std::stringstream ss;
    ss << floats[i];
    size_t w = ss.str().size();
    cellW = std::max(cellW, w);
  }

  int termCols = 80;
  int skipColsSize = (int) iter.cols - (int)(termCols / cellW);
  int skipColsBegin = ((int) iter.cols - skipColsSize) / 2;
  int skipColsEnd = skipColsBegin + skipColsSize;
  if (skipColsEnd <= skipColsBegin - 1) skipColsBegin = -1;

  int termRows = 40;
  int skipRowsSize = (int) iter.rows - (int)(termRows);
  int skipRowsBegin = ((int) iter.rows - skipRowsSize) / 2;
  int skipRowsEnd = skipRowsBegin + skipRowsSize;

  int indent = 0;
//  os << "[";

  for (int matrix = 0; matrix < iter.amount; matrix++) {
    if (iter.rows > 1) {
      if (matrix != 0) {
        os << "],\n";
      }
      os << "[";
      indent++;
    }

    for (int row = 0; row < iter.rows; row++) {
      if (row == 0) {
        os << "[";
      } else {
        os << "]\n";
        os << std::string(indent, ' ') << "[";
      }

      for (int col = 0; col < iter.cols; col++) {
        if (col == skipColsBegin) {
          os << " ... ";
          col = skipColsEnd;
        }
        if (col != 0) os << " ";
        std::stringstream ss;
        float val = floats[matrix * iter.size + row * iter.cols + col];
        ss << val;
        std::string temp = ss.str();
        os << temp << std::string(cellW - temp.size(), ' ');
      }
    }
    os << "]";

    if (t->rank() >= 2) {
      indent--;
    }
  }

  if (iter.rows > 1) {
    os << "]";
  }

}

std::stringstream Print::recorder_;
std::streambuf* Print::coutBuffer_ = nullptr;

void Print::claimCout() {
  coutBuffer_ = std::cout.rdbuf();
  std::cout.rdbuf(recorder_.rdbuf());
}

void Print::unclaimCout() {
  std::cout.rdbuf(coutBuffer_);
}

}
