#pragma once

#include "bits_of_matcha/engine/op/Op.h"

#include <iostream>


namespace matcha::engine::ops {

struct SaveCsv : Op {
  SaveCsv(Tensor* a, const std::string& file);
  static OpMeta<SaveCsv> meta;

  void run() override;

private:

  void dump(std::ostream& os);
  void dumpFrame(std::ostream& os);
  void dumpData(std::ostream& os);

  std::string file_;
};

}
