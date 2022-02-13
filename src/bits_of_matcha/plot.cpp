#include "bits_of_matcha/plot.h"
#include "bits_of_matcha/data.h"
#include "bits_of_matcha/fn/min.h"
#include "bits_of_matcha/fn/max.h"
#include "bits_of_matcha/context.h"


namespace matcha {


Plot::Plot(const Tensor& tensor)
  : tensor_{tensor}
{}

std::ostream& operator<<(std::ostream& os, const Plot& plt) {
  os << plt.string();
  return os;
}

std::string Plot::string() const {
  return asciiHistogram();
}

std::string Plot::asciiHistogram() const {
  Context ctx("Ascii Histogram");
  ctx.debug(false);

  std::string charMap = " `.'^\"-+*=:;!?vcoenxilswmdk#9COD0PS$RB&@XWM";
  int bins = charMap.size();

  size_t size = tensor_.size();
  float* content = tensor_.data();
  auto& shape = tensor_.shape();
  auto& dtype = tensor_.dtype();

  if (dtype != Dtype::Float) throw std::runtime_error("unsupported Dtype");

  size_t width;
  size_t height;
  if (tensor_.rank() > 0) {
    width = shape[-1];
  } else {
    width = 1;
  }
  height = size / width;

  float min, max, binSize;
  if (content) {
    min = fn::min(tensor_).data();
    max = fn::max(tensor_).data();
    binSize = (max - min) / bins;
  }

  std::string buffer;
  buffer.reserve(2 * size + 3 * size / width + 2 * width + 4);

  buffer += "┌";
  for (size_t i = 0; i < 2 * width; i++) buffer += "─";
  buffer += "┐";

  for (size_t i = 0; i < size; i++) {

    char c = '?';
    if (content != nullptr) {
      if (binSize == 0) {
        c = '*';
      } else {
        float val = content[i];
        int bin = (content[i] - min) / binSize;
        bin = std::min(bin, bins - 1);
        c = charMap[bin];
      }
    }

    buffer += c;
    buffer += c;

    if (i % width == 0) {
      if (i != 0) buffer += "│";
      buffer += "\n";
      if (i != height - 1) buffer += "│";
    }
  }

  buffer += "  │";
  buffer += '\n';
  buffer += "└";
  for (size_t i = 0; i < 2 * width; i++) buffer += "─";
  buffer += "┘";

  return buffer;
}


}
