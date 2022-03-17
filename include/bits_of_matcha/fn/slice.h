#pragma once

#include "bits_of_matcha/engine/fn.h"
#include "bits_of_matcha/Shape.h"


namespace matcha::fn {

tensor slice(const tensor& a, const Shape::Slice& slice);
tensor superimpose(const tensor& background, const tensor& foreground, const Shape::Slice& slice);
tensor superimpose(const tensor& background, const tensor& foreground, const Shape::Indices& coords);

}


namespace matcha::engine::fn {

struct SliceIteration {
  SliceIteration(const Shape& source, const Shape::Slice& slice);

  size_t offset;
  size_t continuous;
  std::vector<size_t> strides;
  std::vector<size_t> cycles;
};

class Slice : public Node {
public:
  Slice(Tensor* a, const Shape::Slice& slice);

  void init() override;
  void run() override;

private:
  SliceIteration iter_;

};

}