#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine::ops {

struct LoadCsv : Op {
  LoadCsv(const std::string& file);
  LoadCsv(const std::string& file, const Frame& frame);

  void run() override;

private:
  std::string file_;

  Frame getFrame();

};


}