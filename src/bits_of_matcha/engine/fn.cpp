#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace engine {

Fn::Fn(std::initializer_list<Tensor*> ins)
  : Node(ins)
{}

void Fn::require() {
  requireOuts();
}

void Fn::save(std::ostream& os) const {

}


}
}
