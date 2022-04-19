# Examples

## MNIST classifier


```cpp
using matcha::nn;


int main() {
  matcha::Net classifier {
    nn::FC(100, "relu"),
    nn::FC(50, "relu"),
    nn::FC(10, "softmax")
  };

  classifier.solver = nn::Adam {
    .loss = nn::Crossentropy(),
  };

  auto mnist = matcha::dataset::Csv {
    .file = "my_datasets/mnist.csv",
    .y = {"label"}
  };

  classifier.fit(mnist);
  return 0;
}
```
