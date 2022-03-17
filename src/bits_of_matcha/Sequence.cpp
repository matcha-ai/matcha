#include "bits_of_matcha/Sequence.h"
#include "bits_of_matcha/tensor.h"


namespace matcha {

Sequence::Sequence(std::initializer_list<UnaryFn> phases)
  : phases_{phases}
{}

tensor Sequence::operator()(const tensor& a) {
  tensor feed = a;
  for (auto& phase: phases_) feed = phase(feed);
  return feed;
}

}