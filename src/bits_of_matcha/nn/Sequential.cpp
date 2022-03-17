#include "bits_of_matcha/nn/Sequential.h"
#include "bits_of_matcha/Flow.h"


namespace matcha::nn {


Sequential::Sequential(std::initializer_list<UnaryFn> layers)
  : layers{layers}
{}

void Sequential::fit(const Dataset& dataset) {
  solver(*this, dataset);
}

tensor Sequential::run(const tensor& data) {
  tensor feed = data;
  for (auto& layer: layers) {
    feed = layer(feed);
  }
  return feed;
}

}