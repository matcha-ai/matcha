#include <matcha/tensor>
#include <matcha/engine>

using namespace matcha;

Tensor a(const Tensor& b) {
  flow_init(a, b);

  std::cout << b;
  return b;
}

Tensor softmax(const Tensor& a) {
  Tensor normed = a - fn::max(a);
  Tensor exp = fn::exp(normed);
  return exp / fn::sum(exp);
}

namespace ma = matcha;
namespace mc = matcha;

Tensor relu(const Tensor& a) {
  return fn::max(a, 0);
}

int main() {
  random::Normal rand {
    .m = 3
  };

  while (true) {
    print("---------------------------------");
    Tensor x4 = rand(800, 800);

    x4 = x4.t();
    Tensor x5 = x4.dot(x4);
    x5 = softmax(x5);

    Tensor x6 = fn::product(x5);
    print(x6);
    print(fn::max(x5) + rand() > 3);

    print(engine::stats::memory() >> 10, " kiB");
//    Tensor x5 = x4 + x4;
  }

  return 0;
}