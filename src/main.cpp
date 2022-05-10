#include <matcha/matcha>

using namespace matcha;

auto foo = (Flow) [](tensor digit) {
  auto data = (float*) digit.data();
  for (int i = 0; i < digit.size(); i++) {
    std::cout << data[i] << std::endl;
  }
  print(digit);
  return 0;
};

int main() {
  Dataset ds = dataset::Csv {"mnist_train.csv"};
  tensor digit = ds.get()["x"].reshape(28, 28) / 255;
  foo(digit);
}