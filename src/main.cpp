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

















struct MyNet : Net {
  nn::Fc hidden{200, "relu"};
  nn::Fc output{10, "softmax"};

  tensor run(const tensor& a) override {
    tensor feed = a;
//    feed = hidden(feed);
    feed = output(feed);
    return feed;
  }
};


void testNet() {
//  MyNet net;
  Net net {
    nn::Fc{100, "relu"},
    nn::Fc{.units=10, .flags="softmax"},
  };

  net.loss = nn::Nll{};
//  net.callbacks.clear();

  Dataset mnist = load("mnist_train.csv");
  mnist = mnist.map([](auto& i) { i["x"] /= 255; });
  mnist = mnist.batch(64);
  net.fit(mnist);
}

void testAutograd() {
  tensor b = 0;
  Backprop backprop;

  tensor a = 0;
  tensor y = a + b;

  for (auto&& [t, g]: backprop(y, {&b}))
    print(g, "\n");

  print("y:\n", y, "\n");
}