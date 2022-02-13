#include "bits_of_matcha/fn/argmaxIn.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor argmaxIn(const Tensor& a) {
  auto* node = new engine::fn::ArgmaxIn(a);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}


namespace matcha {
namespace engine {
namespace fn {


ArgmaxIn::ArgmaxIn(Tensor* a)
  : Fn{a}
{
  wrapComputation("ArgmaxIn", {in(0)});
  deduceStatus();
}

ArgmaxIn::ArgmaxIn(const matcha::Tensor& a)
  : ArgmaxIn(deref(a))
{}


}
}
}
