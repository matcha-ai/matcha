#include "bits_of_matcha/nn/Net.h"
#include "bits_of_matcha/Flow.h"


namespace matcha::nn {


Net::Net(const UnaryFn& function)
  : function_{function}
{}

Net::Net(std::initializer_list<UnaryFn> sequential) {
  auto function = [sequential] (tensor feed) {
    for (auto& step: sequential) feed = step(feed);
    return feed;
  };

  function_ = function;
}

void Net::fit(const Dataset& dataset) {
  Context ctx;
  ctx.training = true;

  /*
  std::vector<tensor*> params = pred.params();
  for (auto& t: Ctx::toTrain()) {
    tensor::grad(t);
  }

  solver(function_, dataset);
   */
}

tensor Net::operator()(const tensor& data) {
  return data;
}


const Net::Context* Net::ctx() {
  return Context::current();
}

const Net::Context* Net::Context::current() {
  return current_;
}

Net::Context* Net::Context::current_ = nullptr;

Net::Context::Context() {
  current_ = this;
}


}