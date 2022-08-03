#include <matcha>

using namespace matcha;
using namespace std::complex_literals;


void net();
Dataset dataset();

tensor p;

tensor f(tensor x, tensor w) {
  return matmul(w, x);
}

int main() {
  net(); return 0;
  fn df = grad(f, {1});

  tensor x = ones(2, 5, 1);
  tensor w = ones(3, 5);
  tensor y = f(x, w);
  tensor dx = df(tuple{x, w})[0];

  print(y);
  print(dx);
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

  Dataset mnist = load("mnist_train.csv");
  net.fit(mnist.batch(64), 5);
}