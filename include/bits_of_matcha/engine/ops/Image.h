#pragma once

#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine::ops {

struct Image : Op {
  Image(Tensor* a, const std::string& file);
  static OpMeta<Image> meta;

  void run() override;

private:
  std::string file_;

  void dumpPng();
};

}