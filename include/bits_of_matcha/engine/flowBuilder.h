#pragma once

#include <vector>


namespace matcha::engine {


class Flow;
class Tensor;

class FlowBuilder {
  public:
    FlowBuilder(Flow* flow);
    ~FlowBuilder();

    static FlowBuilder* current();

    void add(Tensor* tensor);
    void inlet(std::vector<Tensor*> tensors);
    void outlet(std::vector<Tensor*> tensors);
    void finish();

  private:
    Flow* flow_;
    std::vector<Tensor*> tensors_;

  private:
    static FlowBuilder* current_;

};

}