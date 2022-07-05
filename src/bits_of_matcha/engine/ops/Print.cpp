#include "bits_of_matcha/engine/ops/Print.h"

#include <iostream>
#include <sstream>


namespace matcha::engine::ops {

Print::Print(Tensor* a, bool endl, std::ostream& os)
  : Op{a}
  , os_(os)
  , endl_(endl)
{
  recorded_ = recorderNext();
}

Print::Print(const std::string& text, bool endl, std::ostream& os)
  : Op{}
  , text_(text)
  , endl_(endl)
  , os_(os)
{
  recorded_ = recorderNext();
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

template <class Type>
inline void dumpScalar(Type scalar, std::ostream& os) {
  if constexpr (std::is_same<Type, bool>()) {
    os << (scalar ? "true " : "false");
  } else if constexpr (
    std::is_same<Type, std::complex<int32_t>>() ||
    std::is_same<Type, std::complex<uint32_t>>() ||
    std::is_same<Type, std::complex<float>>() ||
    std::is_same<Type, std::complex<double>>()
  ) {
    if (std::signbit(scalar.imag())) {
      os << scalar.real() << "-" << -scalar.imag() << "i";
    } else {
      os << scalar.real() << "+" << scalar.imag() << "i";
    }
  } else {
    os << scalar;
  }
}

template <class Type>
void dumpTensorData(Tensor* t, std::ostream& os) {
//  print("buffer is now: ", buffer());
  auto data = t->buffer().as<Type*>();

  if (t->rank() == 0) {
    dumpScalar(data[0], os);
    return;
  }

  auto iter = MatrixStackIteration(t->shape());
  bool oneline = iter.rows == 1;

  size_t cellW = 0;
  if constexpr (std::is_same<Type, bool>()) {
    cellW = 5;
  } else {
    for (size_t i = 0; i < t->size(); i++) {
      std::stringstream ss;
      dumpScalar(data[i], ss);
      size_t w = ss.str().size();
      cellW = std::max(cellW, w);
    }
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
        Type val = data[matrix * iter.size + row * iter.cols + col];
        std::string temp;

        std::stringstream ss;
        dumpScalar(val, ss);
        temp = ss.str();
        temp += std::string(cellW - temp.size(), ' ');
        os << temp;
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

void Print::dumpTensor(std::ostream& os) {
  auto t = inputs[0];

  if (t->frame().null()) {
    os << "[]";
    return;
  }

  if (!t->buffer()) {
    os << t->frame();
    return;
  }

  switch (t->dtype()) {
  case Half: throw std::runtime_error("Half is not supported");
  case Float: dumpTensorData<float>(t, os); break;
  case Double: dumpTensorData<double>(t, os); break;
  case Sbyte: dumpTensorData<int8_t>(t, os); break;
  case Short: dumpTensorData<int16_t>(t, os); break;
  case Int: dumpTensorData<int32_t>(t, os); break;
  case Long: dumpTensorData<int64_t>(t, os); break;
  case Byte: dumpTensorData<uint8_t>(t, os); break;
  case Ushort: dumpTensorData<uint16_t>(t, os); break;
  case Uint: dumpTensorData<uint32_t>(t, os); break;
  case Ulong: dumpTensorData<uint64_t>(t, os); break;
  case Bool: dumpTensorData<bool>(t, os); break;
  case Cint: dumpTensorData<std::complex<int32_t>>(t, os); break;
  case Cuint: dumpTensorData<std::complex<uint32_t>>(t, os); break;
  case Cfloat: dumpTensorData<std::complex<float>>(t, os); break;
  case Cdouble: dumpTensorData<std::complex<double>>(t, os); break;
  default: throw std::runtime_error("invalid dtype");
  }
}

std::stringstream Print::recorder_;
std::streambuf* Print::coutBuffer_ = nullptr;

std::stack<Print::StreamGuard> Print::guards = {};

void Print::claimCout() {
  guards.push({});
  auto& guard = guards.top();
  guard.originalBuffer = std::cout.rdbuf();
  std::cout.rdbuf(guard.recorder.rdbuf());
}

void Print::unclaimCout() {
  auto txt = Print::recorderNext();
  if (!txt.empty()) {
    auto op = new Print(txt, false, std::cout);
    engine::send(op);
  }
  auto& guard = guards.top();
  std::cout.rdbuf(guard.originalBuffer);
  guards.pop();
}

std::string Print::recorderNext() {
  if (guards.empty()) return "";
  auto& recorder = guards.top().recorder;
  std::string buff = recorder.str();
  recorder.str("");
  return buff;
}

}
