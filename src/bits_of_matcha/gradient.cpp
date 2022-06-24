#include "bits_of_matcha/gradient.h"

#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/autograd/gradient.h"

namespace matcha {

std::map<tensor*, tensor> gradient(const std::function<tensor ()>& function,
                                   const std::vector<tensor*>& wrt)
{
  std::vector<engine::Tensor*> wrtInternal;
  wrtInternal.reserve(wrt.size());
  for (auto&& w: wrt)
    wrtInternal.push_back(engine::deref(w));

  auto grads = engine::gradient(function, wrtInternal);

  if (grads.size() != wrt.size())
    throw std::runtime_error("can't pair gradients");

  std::map<tensor*, tensor> result;
  for (int i = 0; i < wrt.size(); i++)
    result[wrt[i]] = engine::ref(grads[i]);

  return result;
}

}