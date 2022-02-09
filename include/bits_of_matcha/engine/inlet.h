#pragma once

#include "bits_of_matcha/engine/node.h"


namespace matcha {
namespace engine {


class Inlet : public Node {
  public:
    Inlet(Tensor* input);

    const Dtype& dtype() const;
    const Shape& shape() const;

    Tensor* openIn() override;
    bool openIn(Tensor* tensor) override;

    static const NodeLoader* loader();
    const NodeLoader* getLoader() const override;
    void save(std::ostream& os) const override;

    bool polymorphicIns() const override;
};


}
}
