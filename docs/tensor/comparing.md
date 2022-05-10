# Comparing

## eq

> `eq(const tensor& a, const tensor& b)` \
> `operator==(const tensor& a, const tensor& b)`

Compares two tensors element-wise. True if elements are equal. Performs broadcasting if necessary. See [add](tensor/basic-arithmetic#add).

## neq

> `neq(const tensor& a, const tensor& b)` \
> `operator!=(const tensor& a, const tensor& b)`

Compares two tensors element-wise. True if elements are not equal. Performs broadcasting if necessary. See [add](#eq).

## max

> `max(const tensor& a, int axis)` \
> `max(const tensor& a)`

Finds the maximum value in given tensor along given axis (negative axis values go from back). The output tensor has rank `a.rank() - 1`.
If no axis is specified, global fold is performed. The output tensor will be a single scalar (i.e. have rank `0`).

## min

> `min(const tensor& a, int axis)` \
> `min(const tensor& a)`

See [max](#max).

## argmax

> `argmax(const tensor& a, int axis)` \
> `argmax(const tensor& a)`

Finds the index of maximum value in given tensor along given axis (negative axis values go from back). The output tensor has rank `a.rank() - 1`.
If no axis is specified, global fold is performed. The output tensor will be a single scalar (i.e. have rank `0`). See [max](#max).

?> NOTE: If the maximum appears in the fold space multiple times, index of first such value is returned.

## argmin

> `argmin(const tensor& a, int axis)` \
> `argmin(const tensor& a)`

See [argmax](#argmax).

## maxBetween

> `maxBetween(const tensor& a, const tensor& b)`

Elementwise maximum of two tensors ([broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) if necessary).

## minBetween

> `min(const tensor& a, const tensor& b)`

See [maxBetween](#maxBetween).
