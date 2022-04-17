#pragma once


namespace matcha::engine {

class Tensor;

class TensorCtx {
public:
  explicit TensorCtx(Tensor* tensor);

  int key() const;

  unsigned mode() const;
  enum {
    Untraced,
    Constant,
    Variable,
  };

private:
  void fixKey(int key);
  void unfixKey();
  bool keyFixed() const;
  void setMode(unsigned mode);

  unsigned mode_;
  int key_;

  friend class Graph;
  friend class Tracer;
  friend class Compiler;
};

}