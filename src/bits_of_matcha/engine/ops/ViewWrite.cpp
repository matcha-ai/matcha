#include "bits_of_matcha/engine/ops/ViewWrite.h"
#include "bits_of_matcha/engine/utils/stdVector.h"


namespace matcha::engine::ops {

ViewWrite::ViewWrite(engine::Tensor* source,
                     engine::Tensor* rhs,
                     const std::vector<engine::Tensor*>& idxs)
  : Op(cat(std::vector<Tensor*>{source, rhs}, idxs))
{
  addOutput(source->frame());
}

Reflection<ViewWrite> ViewWrite::reflection {
.name = "ViewWrite",
};

void ViewWrite::run() {
  outputs[0]->malloc();
}

}
