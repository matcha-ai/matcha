#pragma once

#include "bits_of_matcha/engine/node.h"

#include <vector>


namespace matcha {
namespace engine {

class Inlet;


class Outlet : public Node {
  public:
    Outlet(Tensor* output, const std::vector<Inlet*> inlets);

    const Dtype& dtype() const;
    const Shape& shape() const;

    void eval(unsigned id) const override;

    Tensor* requestOut() override;
    void requestOut(Tensor* tensor) override;

    static const NodeLoader* loader();
    const NodeLoader* getLoader() const override;
    void save(std::ostream& os) const override;

    bool polymorphicOuts() const override;

  private:
    std::vector<Inlet*> inlets_;

};


}
}
