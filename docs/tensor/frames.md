# Tensor frames
> `class Frame`

Frames say how tensor contents should be interpreted. They
provide the underlying scalar datatype ([`Dtype`](#tensor-dtype))
and dimensions ([`Shape`](#tensor-shape)). Usually, frames alone determine,
whether tensors are compatible with some operation as inputs, and how
the output tensors should be eventually framed. Canonically, frames
are represented as `Dtype[Shape]`. For example, `Float[100, 5]` stands 
for a 100x5 matrix of floats. Since scalars have rank `0`, the
brackets are left empty: `Bool[]` for a single boolean.

#### Public methods

- `null() -> bool` - whether `null`, true for uninitialized tensors
- `dtype() const -> const Dtype&` - dtype getter
- `shape() const -> const Shape&` - shape getter
- `bytes() const -> size_t` - total byte size (`shape_size * dtype_size`)
- `string() const -> std::string` - string representation
- `operator==(const Frame& other) const -> bool` - frame equality operator
- `operator!=(const Frame& other) const -> bool` - frame non-equality operator


## Tensor dtype
> `class Dtype`

Immutable underlying tensor datatype:

#### Public methods

- `size() const -> size_t` - number of bytes per one scalar
- `string() const -> std::string` - string representation
- `operator==(const Dtype& other) const -> bool` - dtype equality operator
- `operator!=(const Dtype& other) const -> bool` - dtype non-equality operator

#### Variants

- `Bool` - 8-bit boolean
- `Byte` - 8-bit unsigned integer
- `Ushort` - 16-bit unsigned integer
- `Uint` - 32-bit unsigned integer
- `Ulong` - 64-bit unsigned integer
- `Sbyte` - 8-bit signed integer
- `Short` - 16-bit signed integer
- `Int` - 32-bit signed integer
- `Long` - 64-bit signed integer
- `Half` - 16-bit floating point real
- `Float` - 32-bit floating point real
- `Double` - 64-bit floating point real
- `Cuint` - 2x 32-bit unsigned int complex (64-bit)
- `Cint` - 2x 32-bit signed int complex (64-bit)
- `Cfloat` - 2x 32-bit floating point complex (64-bit)
- `Cdouble` - 2x 64-bit floating point complex (128-bit)

## Tensor shape
> `class Shape`

Immutable multidimensional tensor shape.

#### Public methods

- `rank() const -> size_t` - shape rank, that is, the number of dimensions
- `size() const -> size_t` - the total size "volume" that would fit the entire contents
- `operator[](int idx) -> unsigned` - axis getter, index can be negative to count from the last axis
- `begin() const -> const unsigned*` - the begin iterator
- `end() const -> const unsigned*` - the end iterator
- `operator==(const Shape& other) const -> bool` - shape equality operator
- `operator!=(const Shape& other) const -> bool` - shape non-equality operator

#### Common names by rank

- `rank 0` - scalar
- `rank 1` - vector
- `rank 2` - matrix

