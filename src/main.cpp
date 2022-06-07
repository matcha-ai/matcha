#include <matcha/matcha>

using namespace matcha;

struct : Flow {
  tensor w;

  void init(const tensor& x) {
    print("asdf");
    w = uniform(x.shape());
    requireGrad(&w);
  }

  tensor run(const tensor& x) {
    return x * w;
  }

} foo;

int main() {
  Dataset mnist = dataset::Csv {"mnist_test.csv"};
  print(mnist.size());
  mnist = mnist.map([](Instance i) {
    i["x"] = i["x"].reshape(28, 28) / 255;
    return i;
  });

  for (auto i: mnist.take(5)) {
    tensor x = i["x"];
    x = foo(x);
    print(x != 0, "\n");
  }

}