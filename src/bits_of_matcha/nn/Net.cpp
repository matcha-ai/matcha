#include "bits_of_matcha/nn/Net.h"


namespace matcha::nn {

Net::Net(const AnyOp& op)
  : op_(op)
{}

Net::Net(const std::vector<UnaryOp>& sequence) {
  op_ = [sequence] (tensor feed) {
    for (auto& op: sequence) feed = op(feed);
    return feed;
  };
}

Net::Net(std::initializer_list<UnaryOp> sequence)
  : Net(std::vector(sequence))
{}

void Net::fit(Dataset ds) {
  trainFlow_ = Flow(op_);
  optimizer(ds, trainFlow_);
}

}