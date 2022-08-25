auto testAutograd() -> void;
auto testNet() -> void;

#include <matcha>

using namespace matcha;
using namespace std::complex_literals;

int main() {
//  testNet();
//  testAutograd();

  tensor a = 3;
  tensor b = 2;

  Backprop backprop;

  tensor c = log(b * square(a) + a * 2);
//  tensor c = log(b * square(a));

  // compute the gradients of `c` w.r.t. `a` and `b`
// and return std::map<tensor*, tensor>

  auto gradients = backprop(c, {&a, &b});

  for (auto&& [wrt, gradient]: gradients) {
    std::cout << "the gradient w.r.t. " << wrt << "is ";
    std::cout << gradient << std::endl;
  }

  return 0;
}

















struct MyNet : Net {
  nn::Fc hidden{200, "relu"};
  nn::Fc output{10, "softmax"};

  tensor run(const tensor& a) override {
    tensor feed = a;
    feed = hidden(feed);
    feed = output(feed);
    return feed;
  }
};


void testNet() {
//  Net net {
//    nn::Fc{500, "relu"},
//    nn::Fc{300, "relu"},
//    nn::Fc{10, "softmax"},
//  };
  MyNet net;

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