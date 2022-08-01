#include <matcha>

using namespace matcha;
using namespace std::complex_literals;

void net();
Dataset dataset();

tensor p = 3 * ones(3);

tensor foo(const tensor& x) {
  return x + p;
}

int main() {
  net(); return 0;
  fn f = [](tensor x) {return x;};

  for (int i = 0; true; i++) {

//    auto f = jit(foo);
//    print(info(ones(3, 3)));
//    info(3);
    auto f = jit(foo);
    unary_fn g = jit([&](tensor x) { return f(x);});
    unary_fn h = jit([&](tensor x) { return g(x);});
    print(f(ones(3, 3)));
    p.assign(2);
    print(p);
    print(f(ones(3, 3)));
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
    nn::Fc{400},
    nn::Fc{100, "relu"},
    nn::Fc{50, "relu"},
    nn::Fc{10, "softmax"},
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
  net.fit(mnist.batch(300));
}