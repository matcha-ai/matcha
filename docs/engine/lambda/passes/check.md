# Check
> `engine::check(const Lambda& lambda) -> void`

Passes through the lambda, and performs validity checks. 
Return values:

- `0` - No invalidity found. This is what you usually want.
- `1` - `ops` are not topologically sorted.
- `2` - `ops` are not unique.
- `3` - Some tensors accessible from the lambda are not included in `tensors`.
- `4` - `tensors` are not unique.

## Example

Consider the following function:

```cpp
tensor foo(tensor a) {
  return 3 * a + exp(a);
}
```

Tracing it produces the following lambda:

```txt
lambda(a: Int[]) -> Float[] {
    b = Identity(a)
    c = Cast(b)
    d = Exp(c)
    f = Multiply(e, b)
    g = Cast(f)
    h = Add(g, d)
    i = Identity(h)

    return i
}
```

First, let's check that this lambda is valid.
Indeed, the following code prints `0`:

```cpp
std::cout << check(lambda) << std::endl;
```

Now, suppose we create a custom pass. It will reverse the order of `ops`.
That is of course problematic:

```cpp
void reverse(Lambda& lambda) {
  std::reverse(lambda.ops.begin() lambda.ops.end());
}
```

After running the custom pass, `check` returns `1`, meaning `ops` are not
topologically sorted.


## Op implementation requirements

Check does not query operations on any
[reflection](engine/op/reflection) property.
