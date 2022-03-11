#include "bits_of_matcha/fn/folds.h"


namespace matcha::fn {

Tensor sumAcross(const Tensor& a) {
  auto node = new engine::fn::SumAcross {
    engine::deref(a)
  };

  auto out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor productAcross(const Tensor& a) {
  auto node = new engine::fn::ProductAcross {
    engine::deref(a)
  };

  auto out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor maxAcross(const Tensor& a) {
  auto node = new engine::fn::MaxAcross {
    engine::deref(a)
  };

  auto out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor minAcross(const Tensor& a) {
  auto node = new engine::fn::MinAcross {
    engine::deref(a)
  };

  auto out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor argmaxAcross(const Tensor& a) {
  auto node = new engine::fn::ArgmaxAcross {
    engine::deref(a)
  };

  auto out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor argminAcross(const Tensor& a) {
  auto node = new engine::fn::ArgminAcross {
    engine::deref(a)
  };

  auto out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor sum(const Tensor& a) {
  return sumAcross(a);
}

Tensor product(const Tensor& a) {
  return productAcross(a);
}

Tensor max(const Tensor& a) {
  return maxAcross(a);
}

Tensor min(const Tensor& a) {
  return minAcross(a);
}

Tensor argmax(const Tensor& a) {
  return argmaxAcross(a);
}

Tensor argmin(const Tensor& a) {
  return argminAcross(a);
}

}


namespace matcha::engine::fn {

SumAcross::SumAcross(Tensor* a)
  : Fold{a}
{}

void SumAcross::run() {
  Node::run();
  runCPU(0, std::plus());
}

ProductAcross::ProductAcross(Tensor* a)
  : Fold{a}
{}

void ProductAcross::run() {
  Node::run();
  runCPU(1, std::multiplies());
}

MaxAcross::MaxAcross(Tensor* a)
  : Fold{a}
{}

void MaxAcross::run() {
  Node::run();
  runCPU(
    std::numeric_limits<float>::min(),
    static_cast<const float& (*)(const float&, const float&)>(std::max)
  );
}

MinAcross::MinAcross(Tensor* a)
  : Fold{a}
{}

void MinAcross::run() {
  Node::run();
  runCPU(
    std::numeric_limits<float>::max(),
    static_cast<const float& (*)(const float&, const float&)>(std::min)
  );
}

ArgmaxAcross::ArgmaxAcross(Tensor* a)
  : Fold{a}
{}

void ArgmaxAcross::run() {
  Node::run();

  if (dev_.type == CPU) {

    auto begin = (float*) x_[0]->payload();
    auto end = begin + size_;
    auto& result = *(float*) y_[0]->payload();

    auto max = std::max_element(std::execution::par_unseq, begin, end);
    result = std::distance(begin, max);

  } else {
    throw std::runtime_error("TODO gpu arg*");
  }
}

ArgminAcross::ArgminAcross(Tensor* a)
  : Fold{a}
{}

void ArgminAcross::run() {
  Node::run();

  if (dev_.type == CPU) {

    auto begin = (float*) x_[0]->payload();
    auto end = begin + size_;
    auto& result = *(float*) y_[0]->payload();

    auto min = std::min_element(std::execution::par_unseq, begin, end);
    result = std::distance(begin, min);

  } else {
    throw std::runtime_error("TODO gpu arg*");
  }
}


}