#pragma once

#include "bits_of_matcha/engine/node.h"

#include <functional>


namespace matcha {
  using NullaryFn = std::function<Tensor ()>;
  using UnaryFn   = std::function<Tensor (const Tensor& a)>;
  using BinaryFn  = std::function<Tensor (const Tensor& a, const Tensor& b)>;
  using TernaryFn = std::function<Tensor (const Tensor& a, const Tensor& b, const Tensor& c)>;
}

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
