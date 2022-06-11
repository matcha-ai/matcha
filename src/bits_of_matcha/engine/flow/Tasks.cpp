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

  for (auto& instruction: instructionsForward) {
    instruction();
  }

  std::vector<Tensor*> tempOuts;
  tempOuts.reserve(outputs.size());
  for (int i = 0; i < outputs.size(); i++) {
    Tensor* output = outputs[i];
    Tensor* tempOut = new Tensor(output->frame());
    tempOut->shareBuffer(output);
//    print("HEREEEEEEEEEEE");
//    tempOut->repr(std::cout);
    tempOuts.push_back(tempOut);
  }
  return tempOuts;
}

std::map<tensor*, tensor> Tasks::backward(const std::vector<Tensor*>& ds) {
  if (deltas.empty()) throw std::runtime_error("no backprop flow");
  if (deltas.size() != ds.size()) throw std::invalid_argument("incorrect number of deltas");
  for (int i = 0; i < deltas.size(); i++) deltas[i]->shareBuffer(ds[i]);
  for (auto& instruction: instructionsBackward) instruction();

  std::map<tensor*, tensor> result;
  for (auto [external, target]: grads) {
    auto t = new Tensor(target->frame());
    t->shareBuffer(target);
    result[external] = ref(t);
  }
  return result;
}

}