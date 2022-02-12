#include "bits_of_matcha/fn/fold.h"
#include "bits_of_matcha/fn/add.h"

#include <matcha/engine>
#include <iostream>


namespace matcha {
namespace fn {

Tensor fold(Stream& stream, const Tensor& init, std::function<Tensor (const Tensor& a, const Tensor& b)> fn) {
  stream.reset();

  Tensor output = init.rank() == 0
      ? Tensor(stream)
      : Tensor(init.dtype(), init.shape());

//  if (init.rank() != 0) stream >> output;
  output.subst();
  stream >> output;

  Params buffer(output.dtype(), output.shape(), init);
  Tensor action = fn(buffer, output);

  output.rename("output");
  buffer.rename("buffer");
  action.rename("action");

  int i = 0;
  while (stream) {
    output.update();
    buffer.update(action);
  }

  return buffer;
}

}
}
