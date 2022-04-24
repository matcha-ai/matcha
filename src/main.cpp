#include <matcha/matcha>

using namespace matcha;

tensor p = eye(3, 3);

auto foo = (Flow)[] (const tensor& x) {
  std::cout << "-----------------------" << std::endl;
  std::cout << "inputs " << x.frame() << ":\n" << x << std::endl;
  std::cout << "params " << p.frame() << ":\n" << p << std::endl;
  tensor a = x + 3 * p;
  a += a * a + p + -1 * x;
  a *= 3 + a;
  a = a.pow(.5);
  std::cout << "outputs " << a.frame() << ":\n" << a << std::endl;
  return a;
};

int main() {
  foo.requireGrad(&p);
  tensor x = ones(3, 3);
//  foo(1);
//  foo(2);
//  print(foo(3).pow(2));
  image(uniform(100, 110), "/home/patz/test.png");

  return 0;
}
