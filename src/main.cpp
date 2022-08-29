auto testAutograd() -> void;
auto testNet() -> void;
auto testJit() -> void;

#include <matcha>

using namespace matcha;
using namespace std::complex_literals;

int main() {
//  print(0);
//  testJit();
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
//  Net net = jit([](tensor x) {
//    static auto hidden = nn::Fc{100, "relu"};
//    static auto output = nn::Fc{10, "softmax"};
//    return output(hidden(x));
//  });
  Net net {
//    nn::Fc{100, "relu"},
    nn::Fc{10, "softmax"},
  };

  net.loss = nn::Nll{};
//  net.callbacks.clear();

  Dataset mnist = load("mnist_train.csv");
  mnist = mnist.map([](auto& i) { i["x"] /= 255; });
  mnist = mnist.batch(1);
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

tensor side = 1;

tensor bar(tensor x) {
//  side += 10;
  side *= 2;
  return x;
}

tensor foo(tensor x) {
  return matmul(x, x.t());
}

void testJit() {
  auto joo = jit(foo);

  for (int i = 0; i < 6; i++) {
//    side = i;
    print("------");
//    print(side);
    tensor y = joo(2 * ones(1, 4));
    print(y);
//    print(side);
  }
}