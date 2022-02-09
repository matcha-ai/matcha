#include "bits_of_matcha/fn/fold.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/params.h"
#include "bits_of_matcha/stream.h"
#include "bits_of_matcha/fn/add.h"

#include <iostream>


namespace matcha {
namespace fn {

Tensor fold(const Stream& stream, const Tensor& init, const std::function<Tensor (const Tensor& a, const Tensor& b)>& fn) {
  stream.reset();

  Tensor iter(stream);
  Params buff(iter.dtype(), iter.shape(), init);
  Tensor action = fn(buff, iter);

  iter.rename("iter");
  buff.rename("buff");
  action.rename("action");

  while (stream) {
    iter.update();
    buff.update(action);
  }

  return buff;
}

}
}
