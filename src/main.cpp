#include <matcha/matcha>


class MyFlow : public matcha::Flow {
  tensor p;

  void init(const tensor& a) override {
    p = 3;
  }

  tensor run(const tensor& a) override {
    p += 1;
    return 0;
  }
};

int main() {
  MyFlow myflow;
  tensor x = myflow(3);

  return 0;
}