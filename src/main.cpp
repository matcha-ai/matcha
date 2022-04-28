#include <matcha/matcha>
using namespace matcha;


int main() {
  tensor a = uniform(2, 2);
  tensor b = 2 * eye(6, 2, 2);
  print(a);
  print(b);

  tensor c = a.dot(b);
  print(c);

  return 0;
}
