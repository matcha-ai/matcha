#include "bits_of_matcha/engine/tensor/Tensor.h"

#include <functional>
#include <vector>


namespace matcha::engine {

std::vector<Tensor*> gradient(const std::function<tensor ()>& function, const std::vector<Tensor*>& wrt);

}
