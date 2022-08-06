#include <matcha>

using namespace matcha;
using namespace std::complex_literals;


void net();
Dataset dataset();

tensor w1 = ones(2, 2);

int main() {
  net(); return 0;
}





































Dataset dataset() {
  Dataset mnist = load("mnist_test.csv");
  mnist = mnist.take(1);
  for (int i = 0; i < 2; i++)
    mnist = mnist.cat(mnist);

  return mnist;
}

void net() {
  Net net {
    nn::Flatten{},
//    nn::Fc{1000, "relu"},
    nn::Fc{300, "relu"},
    nn::Fc{100, "relu"},
    nn::Fc{10, "softmax"},
  };

  net.loss = nn::Nll{};
  net.optimizer = nn::Sgd {.lr = 1};
//  net.callbacks.clear();

  Dataset mnist = load("mnist_test.csv");
//  net.step(mnist.batch(64).get());
  net.fit(mnist.batch(64));
}