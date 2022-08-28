# Linear layer
> `struct nn::Linear`

Performs stateful linear (affine) transformation to the inputs
using the layer's `kernel` matrix and `bias` vector:

$ y = K x + b $

?> The kernel initializer (`glorot` by default) can be customized using
   the public member `initializer`, bias is initialized to zeros.


#### Public members

- `unsigned units = 0` - number of units/neurons in the layer
- `bool use_bias = true` - enable/disable bias for the layer
- `Generator initializer = glorot` - kernel initializer function

#### Public methods

- `operator()(const tensor& batch) tensor` - the layer unary operator

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
