#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/flow/graph/Graph.h"

namespace matcha::engine {

class Subflow : public Op {

private:
  Graph graph_;
  std::string name_;

};

OpMeta<Subflow> meta {
  .name = "Subflow"
};

}