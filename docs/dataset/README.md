# Dataset pipelines

Matcha comes together with a seamlessly integrated and modular dataset pipeline system. The pipelines enable you to load (or generate) data
and manipulate it just-in-time or otherwise tune it to your needs.

?> **NOTE:** Use datasets for large amounts of data. Datasets load from the disk only the part that is needed at the moment. \
   This prevents wasting memory.

## Data instances

A single piece of data is called an `Instance`. Instances behave as a dictionary of tensors with string keys. Why? It's quite common to
store more than one information per data entry: for example, we might have labeled images (\*image of a cat\* - "cat", and so on).
For labeled data, it's a convention to have the keys `"x"` for data, `"y"` for label. 
The list of all keys can be obtained by calling the `keys()` method:

```cpp
// we will show this later...
Dataset labeledImages;

// get an instance from the dataset
Instance i = labeledImages.get();

for (auto& key: i.keys()) {
  std::cout << key << ": " << i[key].frame() << std::endl;;
}

//   x: Float[224, 224]
//   y: Float[]

```



## Dataset types and uses

In general, there are two types of pipeline components:

- [Sources](dataset/sources) - load data into pipeline
- [Relays](dataset/relays) - modify data in a pipeline

Let's look at _sources_ first. 
They can load various files from the disk, while using the minimum amount of memory possible.
The following snippet loads the hand-written digits dataset [MNIST](https://en.wikipedia.org/wiki/MNIST_database). 
The training set has in total 60'000 images, 28x28 pixels each. 
However, since every image is represented as a single CSV file row, they are flattened, i.e. they have shape `28 * 28 = 784`.


```cpp
Dataset mnist = dataset::Csv {"mnist_train.csv"};
std::cout << mnist.size() << std::endl;           // 60000

Instance i = mnist.get();
tensor digit = i["x"];
std::cout << digit.dtype() << std::endl;          // Float
std::cout << digit.shape() << std::endl;          // [784]

image(digit, "digit_original.png");

digit = digit.reshape(28, 28);
std::cout << digit.shape() << std::endl;          // [28, 28]

image(digit, "digit_reshaped.png");
```

Contents of `digit_original.png` and `digit_reshaped.png`:

![img](digit_original.png)
![img](digit_reshaped.png)

This is kind of unfortunate. We would like a dataset that provides 28x28 images right away. We can easily do this by
_mapping_ the original dataset. Map is a _relay_ example. It modifies instances of the underlying dataset in a way we want.
Using a lambda function:


```cpp
mnist = mnist.map([](Instance i) {
  i["x"] = i["x"].reshape(28, 28);
  return i;
}
```

This is the magic of pipelines. The new dataset behaves completely the same as the original dataset,
performing all the necessary operations just-in-time for our need. Let's modify it further. The original
dataset pixels have values 0-255. We would like to have them normalized to 0-1:

```cpp
mnist = mnist.map([](Instance i) {
  i["x"] /= 255;
  return i;
}
```

There are too many source and relay datasets to cover all of them here. 
For more, you can read [sources](dataset/sources) and [relays](dataset/relays).


## Reading instances

At last, we would like to get instances from the dataset. We have already seen this way of doing it:

```cpp
Instance i = mnist.get();
```

The method `Dataset::get()` retrieves the next `Instance` from the dataset. We can do that repeatedly to iterate through the entire dataset:

```cpp
while (Instance i = mnist.get()) {
  std::cout << i << std::endl;
}
```

Alternatively, we can use a range-based for loop:

```cpp
for (Instance i: mnist) {
  std::cout << i << std::endl;
}
```

## Jumping through datasets

Datasets behave as linear streams. As such, they provide `tell` method returning the current position,
and `seek` method for changing the current position manually. Note that since the `get` method reads _the next_ Instance,
it increases the position by one. The `reset` method is a shorthand for `seek(0)`.
If the current position is greater or equal to the dataset size, `eof` returns `true`, else it returns `false`.


```cpp
std::cout << mnist.size() << std::endl;     // 60000

mnist.seek(5);
std::cout << mnist.tell() << std::endl;     // 5

mnist.seek(3);
std::cout << mnist.tell() << std::endl;     // 3

mnist.get();
std::cout << mnist.tell() << std::endl;     // 4

std::cout << mnist.eof() << std::endl;      // 0

mnist.seek(60000);
std::cout << mnist.tell() << std::endl;     // 60000

std::cout << mnist.eof() << std::endl;      // 1

mnist.reset();
std::cout << mnist.tell() << std::endl;     // 0
std::cout << mnist.eof() << std::endl;      // 0

mnist.get();
std::cout << mnist.tell() << std::endl;     // 1
std::cout << mnist.eof() << std::endl;      // 0
```

!> **NOTE:** Depending on the concrete pipeline components used, it may or may not be inefficient to jump frequently through the dataset. \
   It's recommended to avoid unneeded jumping and to proceed linearly instead.
