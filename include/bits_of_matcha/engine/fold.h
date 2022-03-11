#pragma once

#include "bits_of_matcha/engine/fn.h"

#include <algorithm>
#include <numeric>
#include <execution>


namespace matcha::engine::fn {


class Fold : public Node {
  public:
    Fold(Tensor* a);
    ~Fold();

    void use(const Device& device) override;
    const Device::Concrete* device() const override;

    template <class BinaryOp>
    inline void runCPU(float init, const BinaryOp& op) {
      auto a = (float*) x_[0]->payload();
      auto b = (float*) y_[0]->payload();

      b[0] = std::accumulate(
        a, a + size_,
        init,
        op
      );
    }

  protected:
    Device::Concrete dev_;
    size_t size_;

};




}
