#include "bits_of_matcha/engine/ops/SideOutput.h"


namespace matcha::engine::ops {

SideOutput::SideOutput(Tensor* source, tensor* target)
  : Op{source}
  , target_(target)
{}

Reflection<SideOutput> SideOutput::reflection {
  .name = "SideOutput",
  .side_effect = true,
};

tensor* SideOutput::target() {
  return target_;
}

void SideOutput::run() {
  auto in = inputs[0];
  auto out = new Tensor(in->frame());
  out->share(in);
//  std::cerr << "assigning to " << target_ << std::endl;
  *target_ = ref(out);
}

}