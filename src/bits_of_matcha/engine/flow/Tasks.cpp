#include "bits_of_matcha/engine/flow/Tasks.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

void Tasks::init() {

}

std::vector<Tensor*> Tasks::forward(const std::vector<Tensor*>& tempIns) {
  for (int i = 0; i < inputs.size(); i++) {
    Tensor* input = inputs[i];
    Tensor* tempIn = tempIns[i];
    input->shareBuffer(tempIn);
  }

  for (Op* op: opsForward) {
//    print("running op ", op, " (", ops::name(op), ")");
    op->run();
  }

  std::vector<Tensor*> tempOuts;
  tempOuts.reserve(outputs.size());
  for (int i = 0; i < outputs.size(); i++) {
    Tensor* output = outputs[i];
    Tensor* tempOut = new Tensor(output->frame());
    tempOut->shareBuffer(output);
    tempOuts.push_back(tempOut);
  }
  return tempOuts;
}

std::vector<Tensor*> Tasks::backward(Tensor* delta) {
  return {};
}

}