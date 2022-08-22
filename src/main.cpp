#include <matcha>

using namespace matcha;
using namespace std::complex_literals;

auto testAutograd() -> void;
auto testNet() -> void;
auto testDataset() -> Dataset;

int main() {
  testNet();
//  testAutograd();
  return 0;
}





































Dataset testDataset() {
  Dataset mnist = load("mnist_train.csv");
  mnist = mnist.take(1);
  for (int i = 0; i < 2; i++)
    mnist = mnist.cat(mnist);

  return mnist;
}

void testNet() {
  Net net {
    nn::flatten,
    nn::Fc{100, "relu"},
    nn::Fc{10, "softmax"},
  };

  net.loss = nn::Nll{};

  Dataset mnist = load("mnist_train.csv");
  mnist = mnist.map([](auto i) { i["x"] /= 255; return i; });
  mnist = mnist.batch(256);
//  for (int i = 0; i < 15; i++)
//    net.step(mnist.get());
  net.fit(mnist);
}

void testAutograd() {
  tensor a = 3*ones(2, 3, 3);
  tensor b = 2*(1 - eye(3, 3));
//  tensor b = 2*ones(3, 3);

  Backprop backprop;

//  tensor normed = b - max(b, -1, true);
//  tensor mapped = exp(b);
//  tensor y = mapped / sum(b, -1, true);
  tensor y = exp(b) - b;

  for (auto&& [t, g]: backprop(y, {&b}))
    print(g, "\n");

  print("y:\n", y, "\n");
}