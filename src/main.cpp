#include <iostream>
#include <matcha/tensor>
#include <matcha/dataset>
#include <matcha/model>
#include <matcha/nn>
#include <matcha/device>


using namespace std;
using namespace matcha;
using namespace matcha::nn;

int main() {
  auto cpu = device::Cpu();
  Context ctx;
  ctx.debug(false);
  ctx.use(cpu);

  Stream s = dataset::csv("/home/patz/Downloads/mnist_train2.csv")
          .batch(1)
          .map([](auto& x) { return x > 0; })
          .map([](auto& x) { return x.reshape({28, 28}); });

  Tensor a = s.fold(0, fn::add);
  cout << a;

  return 0;
}









