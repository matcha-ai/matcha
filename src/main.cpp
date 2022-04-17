#include <matcha/matcha>


auto flow = (matcha::Flow) [](const tensor& a) {
  tensor b = a + 3 + tensor::eye(3, 3) + a;
  auto c = b + 1 + a;
  return c;
};

int main() {
  tensor a = tensor::ones(3, 3);
  flow(a);

  return 0;
}
