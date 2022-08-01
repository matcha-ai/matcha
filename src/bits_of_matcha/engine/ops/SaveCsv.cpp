#include "bits_of_matcha/engine/ops/SaveCsv.h"
#include "bits_of_matcha/engine/tensor/iterations.h"

#include <fstream>
#include <iostream>
#include <sstream>


namespace matcha::engine::ops {

SaveCsv::SaveCsv(Tensor* a, const std::string& file)
  : Op{a}
  , file_(file)
{}

void SaveCsv::run() {
  std::ofstream os(file_);
  dump(os);
}

void SaveCsv::dump(std::ostream& os) {
  dumpFrame(os);
  dumpData(os);
}

void SaveCsv::dumpFrame(std::ostream& os) {
  auto t = inputs[0];

  if (t->dtype() != Float)
    os << "dtype," << t->dtype() << "\n";

  if (t->rank() == 2) return;

  os << "shape";
  for (unsigned dim: t->shape()) os << "," << dim;
  os << "\n";
}

void SaveCsv::dumpData(std::ostream& os) {
  auto t = inputs[0];
  auto f = t->buffer().as<float*>();

  if (t->rank() < 2) {
    float* end = f + t->size();
    for (float* it = f; it != end; it++) {
      if (it != f) os << ",";
      os << *it;
    }
    os << "\n";
    return;
  }

  MatrixStackIteration iter(t->shape());
  float* tensor_end = f + t->size();
  for (float* matrix = f; matrix != tensor_end ; matrix += iter.size) {
    float* matrix_end = matrix + iter.size;
    for (float* row = matrix; row != matrix_end; row += iter.cols) {
      float* row_end = row + iter.cols;
      for (float* col = row; col != row_end; col++) {
        if (col != row) os << ",";
        os << *col;
      }
      os << "\n";
    }
  }

}

}