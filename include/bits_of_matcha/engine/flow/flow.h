#pragma once

#include "bits_of_matcha/ops.h"

#include <memory>
#include <set>

namespace matcha::engine {

class Tensor;
class Op;
class Graph;
class Tasks;

namespace {
std::unique_ptr<Graph> trace(const AnyOp& op);
std::unique_ptr<Graph> autograd(const std::unique_ptr<Graph>& graph);
std::unique_ptr<Tasks> compile(const std::unique_ptr<Graph>& graph);
std::unique_ptr<Tasks> compile(const std::unique_ptr<Graph>& graph, const std::unique_ptr<Graph>& adjoint);
}

}