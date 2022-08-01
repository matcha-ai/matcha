#pragma once

#include "bits_of_matcha/ops.h"

namespace matcha {

  UnaryOp grad(const UnaryOp& function);
//  UnaryOp grad(const BinaryOp & function, const std::index_sequence& wrt);

}