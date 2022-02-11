#pragma once

#include "bits_of_matcha/engine/node.h"


namespace matcha {
namespace device {
  class Computation;
}

namespace engine {

class Fn : public Node {
  public:
    Fn(std::initializer_list<Tensor*> ins);

    void dataStatusChanged(In* in) override;
    void updateStatusChanged(In* in) override;
    void bufferChanged(In* in) override;

    void eval(Out* out) override;
    void prune(Out* out) override;

  protected:
    device::Computation* computation_;

    void wrapComputation(const std::string& name, const std::vector<In*>& ins);
    void deduceStatus();

};


}
}
