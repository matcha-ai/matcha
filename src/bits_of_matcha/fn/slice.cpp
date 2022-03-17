#include "bits_of_matcha/fn/slice.h"
#include "bits_of_matcha/print.h"


namespace matcha::fn {

tensor slice(const tensor& a, const Shape::Slice& slice) {
  auto node = new engine::fn::Slice {
    engine::deref(a),
    slice
  };

  auto out = node->out(0);
  return tensor(out);
}

tensor superimpose(const tensor& background, const tensor& foreground, const Shape::Slice& slice) {
  return 0;
}

tensor superimpose(const tensor& background, const tensor& foreground, const Shape::Indices& coords) {
  return 0;
}

}


namespace matcha::engine::fn {

SliceIteration::SliceIteration(const Shape& source, const Shape::Slice& slice) {
  auto [coords, target] = slice(source);

  offset = 0;
  for (int i = 0; i < source.rank(); i++) {
    offset *= source[i];
    offset += coords[i];
  }

  bool foundBreak = false;
  continuous = 1;
  for (int i = (int) target.rank() - 1; i >= 0; i++) {
    if (foundBreak) {
      strides.push_back(source[i + 1]);
      cycles.push_back(target[i]);
    } else {
      continuous *= target[i];
      print("continuous");
      if (source[i] != target[i]) {
        foundBreak = true;
      }
    }
  }
}


Slice::Slice(Tensor* a, const Shape::Slice& slice)
  : Node{a}
  , iter_(a->shape(), slice)
{
  createOut(a->dtype(), std::get<1>(slice(a->shape())));
}

void Slice::init() {
  Node::init();
}

void Slice::run() {
  Node::run();

  auto a = (float*) x_[0]->payload() + iter_.offset;
  auto b = (float*) y_[0]->payload();

  for (auto cycle: iter_.cycles) std::cout << cycle << " ";
}

}