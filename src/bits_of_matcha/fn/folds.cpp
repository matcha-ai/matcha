#include "bits_of_matcha/fn/folds.h"


namespace matcha::fn {

tensor sum_across(const tensor& a) {
  auto node = new engine::fn::SumAcross {
    engine::deref(a)
  };

  auto out = node->out(0);
  return tensor(out);
}

tensor product_across(const tensor& a) {
  auto node = new engine::fn::ProductAcross {
    engine::deref(a)
  };

  auto out = node->out(0);
  return tensor(out);
}

tensor max_across(const tensor& a) {
  auto node = new engine::fn::MaxAcross {
    engine::deref(a)
  };

  auto out = node->out(0);
  return tensor(out);
}

tensor min_across(const tensor& a) {
  auto node = new engine::fn::MinAcross {
    engine::deref(a)
  };

  auto out = node->out(0);
  return tensor(out);
}

tensor argmax_across(const tensor& a) {
  auto node = new engine::fn::ArgmaxAcross {
    engine::deref(a)
  };

  auto out = node->out(0);
  return tensor(out);
}

tensor argmin_across(const tensor& a) {
  auto node = new engine::fn::ArgminAcross {
    engine::deref(a)
  };

  auto out = node->out(0);
  return tensor(out);
}

tensor sum(const tensor& a) {
  return sum_across(a);
}

tensor product(const tensor& a) {
  return product_across(a);
}

tensor max(const tensor& a) {
  return max_across(a);
}

tensor min(const tensor& a) {
  return min_across(a);
}

tensor argmax(const tensor& a) {
  return argmax_across(a);
}

tensor argmin(const tensor& a) {
  return argmin_across(a);
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