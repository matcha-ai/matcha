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
  Stream s = rng::normal().batch(10);
  Tensor x = floats({3, 3}).subst(s);

  while (s) {
    x.update();
    cout << x;
  }
  Context ctx;
  ctx.use(device::Cpu());


  return 0;
}









