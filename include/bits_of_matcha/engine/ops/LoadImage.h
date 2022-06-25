#pragma once

#include "bits_of_matcha/engine/op/Op.h"

#include <iostream>


namespace matcha::engine::ops {

struct LoadImage : Op {
  LoadImage(const std::string& file);
  LoadImage(const std::string& file, const Frame& frame);

  void run() override;

private:
  std::string file_;

  Frame getFrame(FILE* fp);
  void dumpData(FILE* fp);
};

}