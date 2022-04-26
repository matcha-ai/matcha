#include <matcha/matcha>
using namespace matcha;


tensor weights = 1;
tensor bias = 1;

auto foo = (Flow)[] (const tensor& x) {
  tensor c = weights * x + bias;
  return c;
};

int main() {
  foo.requireGrad({&weights, &bias});

  for (int i = 0; i < 1000; i++) {
    tensor x = uniform(100, 100);
    tensor y = foo(x);
    for (auto& [var, grad]: foo.grad()) {
    }
  }

  return 0;
}
