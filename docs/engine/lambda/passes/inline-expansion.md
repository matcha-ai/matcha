# Inline expansion
> `"bits_of_matcha/engine/lambda/passes/inlineExpansion.h"`\
> `engine::inlineExpansion(Lambda&) -> void`

Finds nested lambdas and recursively inlines them into the outer lambda.

?> This is usually done before other passes 
   to allow further intraprocedural optimizations.

## Example

Suppose the following functions. We will be interested in JITed `foo`, 
that is `joo`:

```cpp
tensor bar(const tensor& a, const tensor& b) {
  return a.t() + b;
}

auto jar = jit(bar);

tensor foo(const tensor& x) {
  return jar(x, x * 3);
}

auto joo = jit(foo);
```

Joo lambda **before the pass**:

```txt
lambda(a: Float[3, 3]) -> Float[3, 3] {
    c = Cast(b)
    d = Multiply(a, c)
    e = Module(a, d)
    f = Identity(e)

    return f
}
```

```plantuml
@startuml
(b) -> (c) : Cast
(<color:blue>**a**) --> (d)
(c) --> (d) : Multiply
(<color:blue>**a**) -> (e)
(d) --> (e) : Module
(e) -> (<color:magenta>**f**) : Identity
@enduml
```

The lambda internally calls a "black box" `Module` operations,
which contains the compiled `bar`/`jar` logic.

Joo lambda **after the pass**:

```txt
lambda(a: Float[3, 3]) -> Float[3, 3] {
    c = Cast(b)
    d = Multiply(a, c)
    e = Identity(a)
    f = Identity(d)
    g = Transpose(e)
    h = Add(g, f)
    i = Identity(h)
    j = Identity(i)
    k = Identity(j)

    return k
}
```

```plantuml
@startuml
(b) -> (c) : Cast
(<color:blue>**a**) --> (d)
(c) --> (d) : Multiply
(<color:blue>**a**) --> (e) : Identity
(d) --> (f) : Identity
(e) --> (g) : Transpose
(g) --> (h)
(f) --> (h) : Add
(h) -> (i) : Identity
(i) -> (j) : Identity
(j) -> (<color:magenta>**k**) : Identity
@enduml
```

The inner `bar`/`jar` module has been inlined into the outer `foo`/`joo` 
lambda using Identity functions. 

We can **simplify this further** by additionally running 
[`engine::copyPropagation`](engine/lambda/passes/copy-propagation):

```txt
lambda(a: Float[3, 3]) -> Float[3, 3] {
    c = Cast(b)
    d = Multiply(a, c)
    e = Transpose(a)
    f = Add(e, d)

    return f
}
```

```plantuml
@startuml
(b) -> (c) : Cast
(<color:blue>**a**) --> (d)
(c) --> (d) : Multiply
(<color:blue>**a**) --> (e) : Transpose
(d) --> (<color:magenta>**f**) : Add
(e) --> (<color:magenta>**f**)
@enduml
```

## Op implementation requirements

All operations in the nested lambdas are expected to be 
[copy-constructible](https://en.cppreference.com/w/cpp/language/copy_constructor). \
Inline expansion does not query operations on any
[`Reflection`](engine/op/reflection) property.

