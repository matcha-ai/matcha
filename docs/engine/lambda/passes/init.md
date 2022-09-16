# Init
> `"bits_of_matcha/engine/lambda/passes/init.h"`\
> `engine::init(Lambda&) -> void`

Initializes all operations inside the lambda in the order 
they are stored in the lambda's `ops`.

!> A lambda should be fully initialized before being passed to an
   [`Executor`](engine/lambda/executors).

## Op implementatin requirements

Init does not query operations on any
[`Reflection`](engine/op/reflection) property. \
Operations can override `init() -> void` for custom initialization logic.
