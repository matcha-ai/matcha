#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/tensor/iterations.h"

#include <iostream>


namespace matcha::engine::ops {

struct Print : Op {
  Print(Tensor* a, bool endl = true, std::ostream& os = std::cout);
  Print(const std::string& text, bool endl = true, std::ostream& os = std::cout);
  static OpMeta<Print> meta;

  void run() override;

  static void claimCout();
  static void unclaimCout();

private:
  std::string recorded_;
  std::string text_;
  std::ostream& os_;
  bool endl_;

  void dumpRecorded(std::ostream& os);
  void dumpTensor(std::ostream& os);
  void dumpText(std::ostream& os);

private:
  static std::stringstream recorder_;
  static std::streambuf* coutBuffer_;
};

}
