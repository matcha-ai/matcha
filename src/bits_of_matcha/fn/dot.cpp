#include "bits_of_matcha/fn/dot.h"
#include "bits_of_matcha/print.h"

#include <cblas.h>


namespace matcha::fn {

tensor dot(const tensor& a, const tensor& b) {
  auto node = new engine::fn::Dot(&a, &b);
  auto out  = node->out(0);
  return tensor(out);
}

}



namespace matcha::engine::fn {


Dot::Dot(const matcha::tensor* a, const matcha::tensor* b)
  : Dot(deref(a), deref(b))
{}

Dot::Dot(Tensor* a, Tensor* b)
  : Node{a, b}
  , a_{a->shape()}
  , b_{b->shape()}
  , dev_{CPU}
{
  if (a_.cols != b_.rows) {
    throw std::invalid_argument("colsA != rowsB");
  }
  if (a->rank() > 2 || b->rank() > 2) {
    if (a->rank() != b->rank()) {
      throw std::invalid_argument("incompatible matrix stacks");
    }
  }
  std::vector<unsigned> dims(a->shape().begin(), a->shape().end());
  dims[dims.size() - 1] = b_.cols;
  dims[dims.size() - 2] = a_.rows;

  createOut(Float, dims);
}

void Dot::run() {
  Node::run();

  if (dev_.type == CPU) {
    auto a = (float*) x_[0]->payload();
    auto b = (float*) x_[1]->payload();
    auto c = (float*) y_[0]->payload();

    for (int i = 0; i < a_.amount; i++) {
      cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans, CblasNoTrans,
        (int) a_.rows, (int) b_.cols, (int) a_.cols,
        1,
        a, (int) a_.cols,
        b, (int) b_.cols,
        0,
        c, (int) b_.cols
      );

      a += a_.size;
      b += b_.size;
      c += a_.rows * b_.cols;
    }

  } else {
    throw std::runtime_error("TODO gpu sgemm");
  }
}

void Dot::use(const Device& device) {
  Computation comp {
    .type = Computation::Dot,
    .cost = a_.rows * b_.cols * a_.cols * a_.amount
  };
  dev_ = device.get(comp);
}

const Device::Concrete* Dot::device() const {
  return &dev_;
}


}