#include <matcha/matcha>


tensor param;

class Foo : public matcha::Flow {
  void init(const tensor& a) override {
    param = 0;
  }

  tensor run(const tensor& a) override {
    return a.dot(a) + param;
  }
};


int main() {
  Foo foo;
  foo(tensor::ones(3, 3));

  foo.grad.add(&param);

//  auto foo = matcha::load("file");

  return 0;
}
