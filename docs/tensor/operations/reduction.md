# Reduction operations

Reduction operations perform a specified function along sequences of
scalars from the input tensor. Each such sequence returns one value in
the output tensor. The sequence can be either the entire input tensor
(global reduction) or, if specified, an axis (axis reduction). When reducing
globally, the output tensor will be a single scalar (rank `0`).
When reducing axis, the tensor rank will be decreased by one. 

?> To preserve the original tensor dimensionality, set the positional 
   argument `keep_dims` to `true`. This will make sure that the output tensor
   can be e.g. broadcasted to the original shape with no extra effort.


## sum
> `sum(const tensor& a, bool keep_dims = false) -> tensor` \
> `sum(const tensor& a, int axis, bool keep_dims = false) -> tensor`

Sums elements:

$\ \sum_{i \in \hat{a}} i $


## max
> `max(const tensor& a, bool keep_dims = false) -> tensor` \
> `max(const tensor& a, int axis, bool keep_dims = false) -> tensor`

Selects maximum from elements:

$\ max \{i \in \hat{a} \} $

## min
> `min(const tensor& a, bool keep_dims = false) -> tensor` \
> `min(const tensor& a, int axis, bool keep_dims = false) -> tensor`

Selects minimum from elements:

$\ min \{i \in \hat{a} \} $

## argmax
> `argmax(const tensor& a, bool keep_dims = false) -> tensor` \
> `argmax(const tensor& a, int axis, bool keep_dims = false) -> tensor`

Selects index of the first maximum from elements:

$\ argmax \{i \in \hat{a} \} $

## argmin
> `argmin(const tensor& a, bool keep_dims = false) -> tensor` \
> `argmin(const tensor& a, int axis, bool keep_dims = false) -> tensor`

Selects index of the first minimum from elements:

$\ argmin \{i \in \hat{a} \} $

## any
> `any(const tensor& a, bool keep_dims = false) -> tensor` \
> `any(const tensor& a, int axis, bool keep_dims = false) -> tensor`

Check whether any from the elements, interpreted as `Bool`, is `true`:

$ \exists i \in \hat{a}: i $

## all
> `all(const tensor& a, bool keep_dims = false) -> tensor` \
> `all(const tensor& a, int axis, bool keep_dims = false) -> tensor`

Check whether all from the elements, interpreted as `Bool`, are `true`:

$ \forall i \in \hat{a}: i $

## none
> `none(const tensor& a, bool keep_dims = false) -> tensor` \
> `none(const tensor& a, int axis, bool keep_dims = false) -> tensor`

Check whether none from the elements, interpreted as `Bool`, is `true`:

$ \nexists i \in \hat{a}: i $

## gather
> `gather(const tensor& a, const tensor& idxs, bool keep_dims = false) -> tensor` \
> `gather(const tensor& a, const tensor& idxs, int axis, bool keep_dims = false) -> tensor`

Gathers elements from `a` according to `idxs`. Indices are counted from `0`
by their position within the current reduction sequence. `idxs` must be `Int` or similar:

$ \hat{a}_ {idxs} $


## mean
> `mean(const tensor& a, bool keep_dims = false) -> tensor` \
> `mean(const tensor& a, int axis, bool keep_dims = false) -> tensor`

Computes the mean value from elements:

$ \overline{\hat{a}} = \frac{1}{|\hat{a}|} \sum_{i \in \hat{a}} i $

## stdev
> `stdev(const tensor& a, bool keep_dims = false) -> tensor` \
> `stdev(const tensor& a, int axis, bool keep_dims = false) -> tensor`

Computes the _biased_ standard deviation from elements:

$ \sqrt{var(\hat{a})} 
= \sqrt{\frac{1}{|\hat{a}|} \sum_{i \in \hat{a}} (i - \overline{\hat{a}})^2} $

## stdevu
> `stdevu(const tensor& a, bool keep_dims = false) -> tensor` \
> `stdevu(const tensor& a, int axis, bool keep_dims = false) -> tensor`

Computes the _unbiased_ standard deviation from elements:

$ \sqrt{var(\hat{a})} 
= \sqrt{\frac{1}{|\hat{a}| - 1} \sum_{i \in \hat{a}} (i - \overline{\hat{a}})^2} $

