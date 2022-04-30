#include <matcha/matcha>
using namespace matcha;

tensor relu(const tensor& a) {
  return maxBetween(a, 0);
}

auto foo = (Flow) [](tensor a) {
  print(a);
  print("------------------------------");
  tensor b = a.t().dot(a);
  print(b);
  return b;
};

int main() {
  tensor a = normal(5, 4, 5);
  tensor b = relu(a);
  a = minBetween(a, 0);

  print(a.shape());
  print(b.shape());
  print(std::string(64, '='));
  print(a);
  print(std::string(64, '='));
  print(b != 0);

  return 0;
}
