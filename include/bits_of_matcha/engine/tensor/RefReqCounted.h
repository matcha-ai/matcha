#pragma once

#include <cinttypes>

namespace matcha::engine {

class RefReqCounted {
public:
  explicit RefReqCounted();
  virtual ~RefReqCounted();

  void ref();
  void unref();
  void req();
  void unreq();

  unsigned refs() const;
  unsigned reqs() const;

private:
  unsigned refs_;
  unsigned reqs_;
};

}