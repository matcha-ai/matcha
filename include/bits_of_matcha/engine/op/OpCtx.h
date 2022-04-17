#pragma once


namespace matcha::engine {

class Op;

class OpCtx {
public:
  explicit OpCtx(Op* op);

  bool traced();
  bool untraced();
  int key() const;

private:
  void fixKey(int key);
  void unfixKey();
  bool keyFixed() const;
  void setTraced();

  int data_;
  friend class Graph;
  friend class Tracer;
  friend class Compiler;
};

}