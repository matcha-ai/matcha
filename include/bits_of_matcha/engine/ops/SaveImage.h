#pragma once

#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine::ops {

struct SaveImage : Op {
  SaveImage(Tensor* a, const std::string& file);
  static Reflection<SaveImage> reflection;

  void run() override;

private:
  std::string file_;

  void dumpPng();
};

}