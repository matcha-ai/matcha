# Automatic differentiation with Matcha

Matcha engine integrates a system for automatically computing _gradients_
(first order or higher order) of given tensors. It works via caching 
relevant operations performed on required tensors
and then [backpropagating](https://en.wikipedia.org/wiki/Backpropagation) 
through them. Note that Matcha backpropagation is fully compatible with 
Matcha function transformations like [JIT](tensor/jit).

?> Automatic differentiation can be be done _dynamically_ on per-instance 
   basis or _statically_ by [JITing](tensor/jit) the backpropagation.
   Static computation features various _performance optimizations_,
   while dynamic computation allows _greater flexibility_ for your code.

!> Higher order derivatives are not fully supported yet.


## Backprop
> `class Backprop`

`Backprop` is the main class for controlling backpropagation. Conceptually, it
roughly maps to [`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape).
Instantiating `Backprop` makes the engine cache all performed operations for
later backpropagation:

```cpp
tensor a = 3;
tensor b = 2;

Backprop backprop;

tensor c = log(b * square(a) + a * 2);
```

Now, we can invoke the backpropagation by calling `backprop` and telling
it what tensor and with respect to (w.r.t.) which tensors to differentiate:

```cpp
// compute the gradients of `c` w.r.t. `a` and `b`
// and return std::map<tensor*, tensor>

auto gradients = backprop(c, {&a, &b});
```

The `gradients` variable now holds a map holding the computed gradients for
$ \frac{\partial c}{\partial a} $ and $ \frac{\partial c}{\partial b} $ .
Let us inspect these by simply iterating through the pairs in the map:

```cpp
for (auto&& [wrt, gradient]: gradients) {
  std::cout << "The gradient w.r.t. " << wrt << " is ";
  std::cout << gradient << std::endl;
}
```

The partial derivative with respect to a `a` and `b` should be:

$
\frac{\partial c}{\partial a} =
\frac{1}{\partial a} log(b a^2 + 2a) =
\frac{2ab + 2}{ba^2 + 2a} =
\frac{2 \cdot 3 \cdot 2 + 2}{2 \cdot 3^2 + 2 \cdot 3} =
\frac{14}{24} = 0.58\overline{3}
$


$
\frac{\partial c}{\partial b} =
\frac{1}{\partial b} log(b a^2 + 2a) =
\frac{a^2}{ba^2 + 2a} =
\frac{3^2}{2 \cdot 3^2 + 2 \cdot 3} =
\frac{9}{24} = 0.375
$


Correct! Our Matcha snippet produces the following output:

```txt
the gradient w.r.t. 0x7ffd2865a468 is 0.583333
the gradient w.r.t. 0x7ffd2865a470 is 0.375
```

## Example


Usually we want to differentiate much larger tensors, such
as matrices of neural network parameters.
The ability to have the gradients computed automatically is priceless
in machine learning. Supposed we have computed the gradients of some
loss function w.r.t. neural network parameters. We can then
perform a single [SGD](nn/optimizers/sgd) step simply like this:

```cpp
std::vector<tensor*> parameters = { ... };
tensor inputs, expected_result;   // suppose we got these from a dataset

Backprop backprop;
tensor outputs = neural_network(inputs);
tensor loss = loss_function(expected_result, outputs);

float learning_rate = 5e-3;

for (auto&& [param, gradient]: backprop(loss, parameters))
  *param -= learning_rate * gradient;
```

For more in-depth explanation of Matcha neural networks, refer to
[this section](nn/).
