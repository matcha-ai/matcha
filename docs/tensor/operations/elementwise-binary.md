# Elementwise binary operations

Elementwise binary operations apply the underlying function individually
to matching pairs of scalars from two input tensors. In the simplest case,
when the two input tensors are of the exactly same `Shape`, this means the
two first elements together, the two second elements, and so on. Else,
[shape broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
is performed. This out-of-the-box empowers operations like scalar 
multplication of a matrix, or addition of a vector to a matrix. When the
inputs are of a different `Dtype`, implicit [dtype promotion]() is performed.


## add
> `add(const tensor& a, const tensor& b) -> tensor` \
> `operator+(const tensor& a, const tensor& b) -> tensor`

Elementwise addition:

$ a + b $

## subtract
> `subtract(const tensor& a, const tensor& b) -> tensor` \
> `operator-(const tensor& a, const tensor& b) -> tensor`

Elementwise subtraction.

$ a - b $

## multiply
> `multiply(const tensor& a, const tensor& b) -> tensor` \
> `operator*(const tensor& a, const tensor& b) -> tensor`

Elementwise multiplication. Also known as the 
[Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)):

$ a \odot b $

## divide
> `divide(const tensor& a, const tensor& b) -> tensor` \
> `operator/(const tensor& a, const tensor& b) -> tensor`

Elementwise (Hadamard) floating-point division:

$ a \oslash b $, $ \frac{a}{b} $

!> The output tensor is always of floating-point representation
   (e.g. `Float`, `Double`, `Cfloat`).

## power
> `power(const tensor& a, const tensor& b) -> tensor`

Elementwise (Hadamard) floating-point power (`pow`) function. The first input is expected to hold
the base scalars, and the second output to hold the exponent scalars.

$ a ^{\odot b} $ , $ a^b $


## eq
> `eq(const tensor& a, const tensor& b) -> tensor` \
> `operator==(const tensor& a, const tensor& b) -> tensor`

Check for elementwise equality. The output is of `Bool` dtype (`true` if equal).

## neq
> `eq(const tensor& a, const tensor& b) -> tensor` \
> `operator==(const tensor& a, const tensor& b) -> tensor`

Check for elementwise non-equality. The output is of `Bool` dtype (`true` if not equal).

## lt
> `lt(const tensor& a, const tensor& b) -> tensor` \
> `operator<(const tensor& a, const tensor& b) -> tensor`

Check elementwise whether less-than. The output is of `Bool` dtype.

## le
> `le(const tensor& a, const tensor& b) -> tensor` \
> `operator<=(const tensor& a, const tensor& b) -> tensor`

Check elementwise whether less-than-or-equal. The output is of `Bool` dtype.

## gt
> `gt(const tensor& a, const tensor& b) -> tensor` \
> `operator>(const tensor& a, const tensor& b) -> tensor`

Check elementwise whether greater-than. The output is of `Bool` dtype.

## ge
> `ge(const tensor& a, const tensor& b) -> tensor` \
> `operator>=(const tensor& a, const tensor& b) -> tensor`

Check elementwise whether greater-than-or-equal. The output is of `Bool` dtype.

## maximum
> `maximum(const tensor& a, const tensor& b)`

Returns elementwise maximum between two scalars from the two input tensors.

## minimum
> `minimum(const tensor& a, const tensor& b)`

Returns elementwise maximum between two scalars from the two input tensors.
