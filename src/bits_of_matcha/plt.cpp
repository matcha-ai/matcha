#include "bits_of_matcha/plt.h"
#include "bits_of_matcha/fn/min.h"
#include "bits_of_matcha/fn/max.h"
#include "bits_of_matcha/context.h"


namespace matcha {


Plt::Plt(const Tensor& tensor)
  : tensor_{tensor}
{}

std::ostream& operator<<(std::ostream& os, const Plt& plt) {
  os << plt.string();
  return os;
}

std::string Plt::string() const {
  return asciiHistogram();
}

std::string Plt::asciiHistogram() const {
  Context ctx("Ascii Histogram");
  ctx.debug(false);

  std::string charMap = " `.'^\"-+*=:;!?vcoenxilswmdk#9COD0PS$RB&@XWM";
  int bins = charMap.size();

  size_t size = tensor_.size();
  float* content = (float*)tensor_.data();
  auto& shape = tensor_.shape();
  auto& dtype = tensor_.dtype();

  if (dtype != Dtype::Float) throw std::runtime_error("unsupported Dtype");

  size_t width;
  if (tensor_.rank() > 0) {
    width = shape[-1];
  } else {
    width = 1;
  }

  float min, max, binSize;
  if (content) {
    min = *(float*)fn::min(tensor_).data();
    max = *(float*)fn::max(tensor_).data();
    binSize = (max - min) / bins;
  }

  std::string buffer;
  buffer.reserve(2 * size + size / width);

  for (size_t i = 0; i < size; i++) {
    if (i % width == 0 && i != 0) {
      buffer += '\n';
    }

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
  }

  buffer += '\n';
  return buffer;
}


}
