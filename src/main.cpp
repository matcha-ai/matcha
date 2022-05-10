#include <matcha/matcha>

using namespace matcha;


int main() {
  Dataset ds = dataset::Csv {"mnist_train.csv"};
  std::cout << ds.size() << std::endl;
  ds = ds.take(2);

  while (Instance i = ds.get()) {
  }
}