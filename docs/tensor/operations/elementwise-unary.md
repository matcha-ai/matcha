# Elementwise unary operations

Elementwise unary operations apply the underlying function individually 
to each scalar element of the input input tensor to produce the output 
tensor. The input can be of any shape. The output will have the same shape.


## cast
> `cast(const tensor& x, const Dtype&) -> tensor`

Casts tensor to given `Dtype`.

!> Some type conversions can be implementation-defined.

## positive
> `positive(const tensor& x) -> tensor` \
> `operator+(const tensor& x) -> tensor`

Elementwise scalar positive. Equivalent to elementwise multiplication by `+1`.

!> Passive in most cases.
   However, inputs of type `Bool` are converted to `Byte`, holding either `0` or `1`.

## negative
> `negative(const tensor& x) -> tensor` \
> `operator-(const tensor& x) -> tensor`

Elementwise scalar negative. Equivalent to elementwise multiplication by `-1`.
See [positive](#positive).

## exp
> `exp(const tensor& x) -> tensor`

Elementwise natural exponential function:

$ e^x $

## log
> `log(const tensor& x) -> tensor`

Elementwise natural logarithm function:

$ log{x} $ 

## square
> `square(const tensor& x) -> tensor`

Elementwise square:

$ x^2 $

## sqrt
> `sqrt(const tensor& x) -> tensor`

Elementwise square root: 

$ \sqrt{x} $

## sigmoid
> `sigmoid(const tensor& x) -> tensor`

Elementwise [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function:

$ \sigma(x) = \frac{1}{1 + e^{-x}} $

## tanh
> `tanh(const tensor& x) -> tensor`

Elementwise hyperbolic tangent function; equivalent to:

$ 2 \sigma(2 x) - 1$
