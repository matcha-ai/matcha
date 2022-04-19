#include <matcha/matcha>


tensor params = tensor::ones(3, 3);
tensor another = 3;
tensor more = 0;

auto flow = (matcha::Flow) [](const tensor& a) {
  tensor b = a + another + params.dot(a);
  auto c = b + a;
  return (c.t().dot(params) + more).t();
};

int main() {
  flow.requireGrad(another);
  flow.requireGrad(params);
  flow.requireGrad(more);
  tensor a = tensor::ones(3, 3);
  flow(a);


  return 0;
}
