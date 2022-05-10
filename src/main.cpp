#include <matcha/matcha>

using namespace matcha;

tensor weights = normal(2, 2);

auto foo = (Flow) [](tensor x) {
  tensor a = weights.dot(x);
  a = square(a) + exp(a) + 2;
  tensor b = a.t().dot(x);
  b += a * x;
  std::cout << b << std::endl;
  return b;
};

int main() {
  tensor x = eye(2, 2);
  foo(x);
  foo(2 * x);
}