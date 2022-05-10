#include <matcha/matcha>

using namespace matcha;


int main() {
  Dataset ds = dataset::Csv {"/home/patz/Downloads/mnist_train.csv"};
  ds = ds.map([] (Instance i) {
    i["x"] *= .1 * normal(i["x"].shape());
    return i;
  });

  Net net {
    nn::Fc{30, "relu"},
    nn::Fc{30, "softmax"},
  };

  net.fit(ds);
}