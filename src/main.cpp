#include <matcha>

using namespace matcha;
using namespace std::complex_literals;

void net();
Dataset dataset();

tensor p;

tensor foo(const tensor& x) {
  p = x*x;
  return p;
}

int main() {
  net(); return 0;
  unary_fn f = jit(foo);
  unary_fn g = jit([=](tensor x) { return 2 * f(x); });

  for (int i = 0; true; i++) {
    print(p);
    print("------");
    tensor x = ones(3, 3);
    tensor y = g(x);
    print(x, "\n", y);
    print("------");
    print(p);

    break;
  }
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
//    nn::Fc{400},
//    nn::Fc{100, "relu"},
//    nn::Fc{50, "relu"},
    nn::Fc{10, "nobias"},
  };

  net.loss = mse;
//  net.callbacks.clear();

//  Dataset mnist = load("mnist_train.csv");
  Dataset mnist = (Dataset) []() {
    Instance i;
    i["x"] = ones(28, 28);
    i["y"] = cast(1, Int).reshape(1, 1);
    return i;
  };
  mnist = mnist.take(1000);
  mnist = mnist.cat(mnist).cat(mnist).cat(mnist);
  mnist = load("mnist_train.csv");
  net.fit(mnist.batch(64));
}