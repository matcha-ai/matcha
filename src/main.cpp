#include <matcha>
#include <fstream>

using namespace matcha;
using namespace std::complex_literals;

tensor foo(tensor a, tensor b) {
  tensor pl = a + b;
  tensor mi = a - b;
  tensor mu = a * b;
  tensor di = a / b;

  pl = pl.t();
  mi = mi.t();
  mu = mu.t();
  di = di.t();

  tensor c = matmul(pl, mi);

  tensor d = 2.71828;
  d *= -1i;

  return c * d;
}

int main() {
  std::vector frames = { Frame() };
  auto foo = jit(matcha::sigmoid);
  std::cout << foo(ones(3, 3)) << std::endl;
}
