# Examples

## MNIST classifier


```cpp
#include <matcha/matcha>
using namespace matcha;


int main() {
  // create neural net

  Net classifier {
    nn::Flatten {},
    nn::Fc {100, "relu"},
    nn::BatchNorm {},
    nn::Fc {50, "relu"},
    nn::BatchNorm {},
    nn::Fc {10, "softmax"}
  };

  classifier.optimizer = nn::Adam {
    .loss = nn::Crossentropy(),
  };


  // create dataset

  Dataset mnist = dataset::Csv {
    .file = "mnist_train.csv",
    .y = {"label"}
  };

  mnist = mnist.map([](Instance i) {
    i["x"] /= 255;
    return i;
  });


  // train!

  classifier.fit(mnist);
}
```
