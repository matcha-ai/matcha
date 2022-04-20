#define MATCHA_GENERATOR_TAIL() \
                 \
public:                 \
const Generator generator = init();                    \
                    \
tensor operator()(const Shape& shape) { \
  return generator(shape);                    \
}\
\
template <class... Dims>      \
inline tensor operator()(Dims... dims) {      \
  return operator()(VARARG_SHAPE(dims...));     \
}                   \
operator Generator() {                        \
  return generator;                   \
}
