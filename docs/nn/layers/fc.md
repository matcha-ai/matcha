# Fully connected layer
> `struct nn::Fc`

The fully connected layer wraps [`nn::Linear`](nn/layers/linear) and
various activation functions and other configurable adjustments.

#### Public members

- `unsigned units = 0` - number of layer units/neurons
- `std::string flags = ""` - additional configurations

#### Public methods

- `operator()(const tensor& batch) tensor` - the layer unary operator

#### Internal logic

- `init() - Layer()` -
  parses configuration and initializes the internal layer implementation
- `internal_{init()} -> std::shared_ptr<Layer>` -
  the internal implementation

!> The internal logic should not be modified directly. It is exposed
   to enable the C++20 [aggregate initialization](https://en.cppreference.com/w/cpp/language/aggregate_initialization)
   feature.

## Construction syntax

The `Fc` struct is designed in a way that enables simple and readable
initialization. Using the C++20 [aggregate initialization](https://en.cppreference.com/w/cpp/language/aggregate_initialization)
syntax:

```cpp
auto layer = nn::Fc{.units=200, .flags="relu"};
```

Or for short:

```cpp
auto layer = nn::Fc{200, "relu"};
```

This creates a fully connected layer with 200 units and the Rectified
Linear Unit (ReLU) activation function.

## Configuration options

The configuration is deduced from the public member `flags` as follows:

- Separate individual flags with a single comma (`,`)
- Any whitespace characters around commas are ignored
- The flags parser is case-insensitive
- The flags are searched for common activation functions; the last
  recognized activation function is used. Supported functions are:
  - [`relu`](nn/layers/relu), 
    [`tanh`](nn/layers/tanh),
    [`sigmoid`](nn/layers/sigmoid),
    [`softmax`](nn/layers/softmax), 
    [`exp`](nn/layers/exp)
  - For no activation, either don't use any activation flag,
    or explicitly specify one of `none`, `identity`, `id`
- The `nobias` flag can be used to disable lienar layer bias
- For batch normalization, use one of `bn`, `bnorm`, `bnormalize`,
  `bnormalization`, `batchnorm`, `batchnormalize`, `batchnormalization`
  - Using batch normalization automatically **disables linear layer bias**
    and pushes the layer activation function **after** batch normalization
  - **Note:** batch normalizatin is still work in progress

## Code style recommendations and examples

- Don't separate the constructor body from the layer type `Fc`
  - rather, use the `auto` keyword
- Keep the flags lowercase
- Use shorter flag variants (like `bn`)
- For no activation, either leave `flags` empty or use `none`


```cpp
auto hidden1 = nn::Fc{100, "relu,bn"};
auto hidden2 = nn::Fc{100, "tanh, nobias"};
auto categorical_classification = nn::Fc{10, "softmax"};
auto binary_classification = nn::Fc{1, "sigmoid"};
auto poisson_regression = nn::Fc{1, "exp"};
```
