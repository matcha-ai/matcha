# Basic arithmetic

## add

> `add(const tensor& a, const tensor& b)` \
> `operator+(const tensor& a, const tensor& b)` 

Adds two tensors together elementwise. The input tensors must either have the same shape or one of them has to be a scalar.

```cpp
tensor a = tensor::ones(3, 3);
tensor b = tensor::ones(4, 4);
tensor c = tensor::eye(3, 3);
tensor d = 3;

tensor e; 
e = a + b;  // error: incompatible shapes
e = a + c;  // OK
e += d;     // OK
```

## subtract

> `subtract(const tensor& a, const tensor& b)` \
> `operator-(const tensor& a, const tensor& b)`

Subtracts two tensors elementwise. The compatibility rules copy [`add`](#add).

## negative

> `negative(const tensor& a)` \
> `operator-(const tensor& a)`

Elementwise negative.

## abs

> `abs(const tensor& a)`

Elementwise abs.

## multiply

> `multiply(const tensor& a, const tensor& b)` \
> `operator*(const tensor& a, const tensor& b)`

Multiplies two tensors elementwise. The compatibility rules copy [`add`](#add).

## divide

> `divide(const tensor& a, const tensor& b)` \
> `operator/(const tensor& a, const tensor& b)`

Divides two tensors elementwise. The compatibility rules copy [`add`](#add).

## pow

> `pow(const tensor& a, const tensor& b)` \
> `tensor::pow(const tensor& b)`

Elementwise power. The compatibility rules copy [`add`](#add).

## square

> `square(const tensor& a)` 

Elementwise square.

## sqrt

> `sqrt(const tensor& a)` 

Elementwise square root.

## exp

> `exp(const tensor& a)` 

Elementwise natural exponential.

## log

> `log(const tensor& a)` 

Elementwise natural logarithm.
