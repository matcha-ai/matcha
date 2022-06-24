#pragma once

#include "bits_of_matcha/tensor.h"

#include <map>
#include <vector>
#include <functional>


namespace matcha {

std::map<tensor*, tensor> gradient(const std::function<tensor ()>& function, const std::vector<tensor*>& wrt);

}