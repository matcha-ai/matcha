auto testAutograd() -> void;
auto testNet() -> void;

#include <matcha>

using namespace matcha;
using namespace std::complex_literals;

int main() {
  testNet();
//  testAutograd();
  return 0;
}



















void testNet() {
  Net net {
    nn::Fc{500, "relu"},
    nn::Fc{300, "relu"},
    nn::Fc{10, "softmax"},
  };

  net.loss = nn::Nll{};
//  net.callbacks.clear();

  Dataset mnist = load("mnist_train.csv");
  mnist = mnist.map([](auto& i) { i["x"] /= 255; });
  mnist = mnist.batch(64);
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