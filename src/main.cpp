#include <matcha/matcha>

using namespace matcha;

int main() {
  std::vector<float> v {1, 2, 3, 55};
  tensor x = blob(v.data(), {4});
}