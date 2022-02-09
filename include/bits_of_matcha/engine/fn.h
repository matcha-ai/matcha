#pragma once

#include "bits_of_matcha/engine/node.h"


namespace matcha {
namespace engine {

class Fn : public Node {
  public:
    Fn(std::initializer_list<Tensor*> ins);

    void require() override;
    void save(std::ostream& os) const override;

};


}
}
