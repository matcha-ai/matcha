# Buffer

> `engine::Buffer`

Wrapper for an optional contiguous block of memory somewhere, on some device.

## Constructors

- `explicit Buffer()` - an empty buffer
- `explicit Buffer(size_t bytes)` - a buffer guaranteed to fit required amount of bytes
- `Buffer(const Buffer& other)` - buffer that shares memory with `other`
- `Buffer(Buffer&& other)` - move constructor, `other` is freed

## Data management
- `malloc(size_t bytes) -> void` - allocates at least `bytes` bytes
- `free() -> void` - frees its payload if allocated
- `operator=(const Buffer& other) -> Buffer&` - shares payload from `other`
- `operator=(Buffer&& other) -> Buffer&` - takes payload from `other`
- `operator==(const Buffer& other) -> bool` - true iff the two buffers share payload
- `operator!=(const Buffer& other) -> bool` - true iff the two buffers do not share payload
- `bytes() const -> size_t` - get allocated bytes, might be greater than required size

## Data access

- `payload() -> void*` - retrieves internal payload, e.g. RAM pointer
- `template <class T> as() -> T` - retrieves payload casted to `T`
