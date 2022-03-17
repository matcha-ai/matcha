#include "bits_of_matcha/fn/transpose.h"
#include "bits_of_matcha/print.h"

#include <cblas.h>


namespace matcha::fn {

tensor transpose(const tensor& a) {
  auto node = new matcha::engine::fn::Transpose {
    engine::deref(a)
  };

  auto out  = node->out(0);
  return tensor(out);
}

}


namespace matcha::engine::fn {


Transpose::Transpose(Tensor* a)
  : Node{a}
  , a_{a->shape()}
  , dev_{CPU}
{
  std::vector<unsigned> dims(a->shape().begin(), a->shape().end());
  if (dims.size() == 1) {
    dims.resize(2);
  }
  dims[dims.size() - 1] = a_.rows;
  dims[dims.size() - 2] = a_.cols;

  if (dims.size() == 2 && dims[1] == 1) {
    dims.erase(dims.end() - 1);
  }

  if (a_.rows == 1 || a_.cols == 1) {
    idle_ = true;
  } else {
    idle_ = false;
  }
  createOut(Float, dims);
}

void Transpose::init() {
  if (in(0)->source()) in(0)->source()->init();
  if (idle_) {
    out(0)->shareBuffer(in(0));
  } else {
    Node::init();
  }
}

void Transpose::run() {
  Node::run();

  if (idle_) return;

  if (a_.rows == 1 || a_.cols == 1) {
    return;
  }

  if (dev_.type == CPU) {

    auto a = (float*) x_[0]->payload();
    auto b = (float*) y_[0]->payload();

    for (int matrix = 0; matrix < a_.amount; matrix++) {

      auto begA = a;
      auto begB = b;

      for (int i = 0; i < a_.rows; i++) {
        for (int j = 0; j < a_.cols; j++) {
          b[j * a_.rows + i] = a[i * a_.cols + j];
        }
      };
    }


  } else {
    throw std::runtime_error("TODO gpu transpose");
  }
}

void Transpose::use(const Device& device) {
  Computation comp {
    .type = Computation::Transpose,
    .cost = in(0)->size()
  };
  dev_ = device.get(comp);
}

const Device::Concrete* Transpose::device() const {
  return &dev_;
}


}