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
    return x - w * w + w;
  }

} foo;

auto bar = (Flow) [](tensor x) {
  float a = 1;
  return a * x + 0 * foo(x);
};

int main() {
  Dataset mnist = load("mnist_test.csv");
  tensor mari = load("mari.jpeg");
  mnist = mnist.map([](Instance i) {
    i["x"] = i["x"].reshape(28, 28) / 255;
    return i;
  });

  for (auto i: mnist.take(5)) {
    tensor x = i["x"];
    print(x != 0, "\n");
  }


}