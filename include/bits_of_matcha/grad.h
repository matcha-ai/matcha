#pragma once

#include "bits_of_matcha/fn.h"

namespace matcha {

fn grad(const fn& function, const std::vector<int>& argnum = {0});
fn value_and_grad(const fn& function, const std::vector<int>& argnum = {0});

}